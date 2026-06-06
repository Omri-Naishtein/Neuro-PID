import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped
from Neuro import OnlineTuner
import os
import csv

# ── GPIO CONFIG ───────────────────────────────────────────────────────────────
CHIP     = "gpiochip0"
ENA_LINE = 41
ENB_LINE = 43
IN1_LINE = 53
IN2_LINE = 113
IN3_LINE = 52
IN4_LINE = 51

# ── PARAMETERS (from working version) ────────────────────────────────────────
PWM_MAX        = 75       # motor cap (%)
CRUISE_PWM     = 75       # max forward speed (allow fuller throttle at mid-range)
MIN_PWM        = 32       # minimum motor speed — raised so robot doesn't crawl slowly

# ── FORWARD OBSTACLE THRESHOLDS ──────────────────────────────────────────────
STOP_DIST          = 0.20  # m — stop and enter WAIT when front is closer than this
DEADEND_SIDE_DIST  = 0.20  # m — both sides must be below this to trigger a U-turn
CONFIRM_SCANS      = 2     # consecutive scans below STOP_DIST before stopping
WAIT_SECONDS       = 2.0   # s — pause after stopping before entering DECIDE
SETTLE_SECONDS     = 1.5   # s — hold position after a turn before resuming forward

# ── TURN PARAMETERS ───────────────────────────────────────────────────────────
TURN_ANGLE_DEG   = 90.0   # degrees for a standard 90° turn
TURN_DONE_DEG    = 1.0    # degrees — completion tolerance (tighter = more exact)
TURN_MIN_PWM     = 20     # % — motor floor; ramps down near target to prevent overshoot
TURN_TIMEOUT     = 5.0    # s — give up and go forward if turn stalls
DEADEND_TURN_DEG = 180  
TURN_DEADBAND    = 0.10   # m — left/right must differ by more than this to turn;
                           #     smaller differences are treated as equal (noise guard)

IMU_CALIB_TIME = 5.0      # s — gyro bias calibration window

# ── LIDAR CONE WIDTHS AND RANGE (all easy to tune here) ──────────────────────
FRONT_CONE_DEG = 5.0      # ±° half-angle of the front danger cone
SIDE_CONE_DEG  = 5.0      # ±° half-angle of each side sensing cone
MAX_RANGE      = 2.0      # m — discard returns beyond this distance

# deadband for gyro rate (deg/s)
GYRO_DEADBAND  = 1.0

# ── SAFE GAINS (used as MLP init / fallback) ─────────────────────────────────
HEAD_KP, HEAD_KI, HEAD_KD = 2.0, 0.15, 0.05    # heading baseline (MLP init)
TURN_KP, TURN_KI, TURN_KD = 2.0, 0.3,  0.48    # turn baseline (MLP init)
# Distance → speed gains come from the `Neuro.OnlineTuner` MLP (safe default ~= KP=80, KI=0.5, KD≈17.9)

# ── TUNING: zeta-band and speed feedforward --------------------------------
# Wider zeta bands make the tuner less likely to instantly revert to safe gains.
ZETA_BAND_HEAD = 0.5
ZETA_BAND_DIST = 0.6
ZETA_BAND_TURN = 0.5

# Feedforward blending for distance→PWM: weight in [0,1] (0 => only PID,
# 1 => only feedforward). EXP < 1 boosts mid-range speeds.
SPEED_FF_WEIGHT = 0.45
SPEED_FF_EXP = 0.6

# Runtime logging
DEBUG_LOG = True
LOG_FILENAME = "tuner_log.csv"


# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpio_set(line, value):
    subprocess.run(["gpioset", CHIP, f"{line}={value}"], check=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    """Wrap to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0

def wrap_rad(a):
    """Wrap angle to [-pi, pi] radians."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class NeuralPID:
    """Stateful PID integrator whose gains are supplied externally each step.
    Used with Neuro.OnlineTuner: the tuner (an MLP) provides (kp, ki, kd) and
    this class only maintains the integral / derivative state and applies them.
    There are NO hand-tuned constant-gain PIDs anywhere — every loop is neural."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.integral  = 0.0
        self.prev_err  = 0.0
        self.prev_time = None

    def step(self, error, kp, ki, kd):
        now = time.monotonic()
        dt  = 0.02 if self.prev_time is None else max(1e-3, now - self.prev_time)
        self.prev_time = now
        self.integral  = clamp(self.integral + error * dt, -1.0, 1.0)
        deriv          = (error - self.prev_err) / dt
        self.prev_err  = error
        return kp * error + ki * self.integral + kd * deriv


# ── SOFTWARE PWM ──────────────────────────────────────────────────────────────
class SoftPWM:
    def __init__(self, line, freq=100):
        self.line, self.freq, self.duty = line, freq, 0
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        period = 1.0 / self.freq
        while self.running:
            if self.duty <= 0:
                gpio_set(self.line, 0); time.sleep(period)
            elif self.duty >= 100:
                gpio_set(self.line, 1); time.sleep(period)
            else:
                gpio_set(self.line, 1); time.sleep(period * self.duty / 100)
                gpio_set(self.line, 0); time.sleep(period * (1 - self.duty / 100))

    def set(self, duty):
        self.duty = clamp(duty, 0, 100)

    def stop(self):
        self.running = False
        gpio_set(self.line, 0)


# ── ROBOT NODE ────────────────────────────────────────────────────────────────
class Robot(Node):
    def __init__(self):
        super().__init__("robot")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(LaserScan, "/scan", self.scan_cb, qos)
        self.create_subscription(Vector3Stamped, "/imu/euler_angles", self.imu_cb, 10)

        self.left_pwm  = SoftPWM(ENA_LINE)
        self.right_pwm = SoftPWM(ENB_LINE)

        # ── gyro state ────────────────────────────────────────────────────
        self.prev_raw_yaw  = None
        self.prev_raw_time = None
        self.calib_rates   = []
        self.drift_offset  = 0.0
        self.heading       = 0.0
        self.imu_ready     = False

        # ── lidar distances (three cones, updated every scan) ─────────────
        self.dist_front = float("inf")
        self.dist_left  = float("inf")
        self.dist_right = float("inf")

        self.cone_front = math.radians(FRONT_CONE_DEG)
        self.cone_side  = math.radians(SIDE_CONE_DEG)

        # ── state machine ─────────────────────────────────────────────────
        # INIT -> FORWARD -> WAIT -> DECIDE -> TURN -> SETTLE -> FORWARD  (repeats)
        self.state           = "INIT"
        self.start_heading   = 0.0
        self.turn_target     = 0.0
        self.turn_started_at = 0.0
        self.pause_until     = 0.0
        self.settle_until    = 0.0
        self.obstacle_count  = 0
        self.start_time      = time.monotonic()

        # ── Neural controllers (Neuro.OnlineTuner) ───────────────────────
        # Every loop is a PID whose (Kp, Ki, Kd) are produced online by a small
        # MLP — there are NO hand-tuned constant-gain PIDs.  Each tuner drives
        # its damping estimate  ζ = Kd / (2·√(mass·Kp))  toward a target; the
        # `mass` scale lets the same network design serve loops whose gain
        # magnitudes differ by orders of magnitude.  Gains START at the values
        # below (the safe fallback) and the MLP only learns deviations.

        # Heading — keep the robot driving straight.  mass≈6e-4 puts ζ=1 on the
        # small heading gains; output is steering trim (±20%).
        self.head_tuner = OnlineTuner(
            train=True,
            kp_range=(0.5, 8.0), ki_range=(0.0, 1.0), kd_range=(0.0, 0.5),
            init_gains=(HEAD_KP, HEAD_KI, HEAD_KD),
            mass=0.0003, target_zeta=0.8, zeta_band=ZETA_BAND_HEAD,
            feat_stop=0.0, feat_scale=90.0, feat_pwm=20.0,
            clamp_near_stop=False,
        )
        self.head_pid   = NeuralPID()
        self._last_corr = 0.0

        # Distance → speed — the wall-approach loop.  mass=1.0, ζ=1 (critically
        # damped: fastest approach with no overshoot into the wall).
        self.dist_tuner = OnlineTuner(
            train=True,
            feat_stop=STOP_DIST, feat_scale=MAX_RANGE, feat_pwm=float(PWM_MAX),
            target_zeta=0.8, zeta_band=ZETA_BAND_DIST,
        )
        self.dist_pid    = NeuralPID()
        self._last_speed = float(MIN_PWM)

        # Turn — rotate to a target heading. We intentionally target an
        # underdamped controller (ζ≈0.7) to make turns faster; SETTLE absorbs
        # the small overshoot. Allow faster online gain changes (higher
        # `max_rate`) so the tuner can respond aggressively during a turn.
        self.turn_tuner = OnlineTuner(
            train=True,
            kp_range=(0.5, 8.0), ki_range=(0.0, 1.0), kd_range=(0.0, 2.0),
            init_gains=(TURN_KP, TURN_KI, TURN_KD),
            mass=0.06, target_zeta=0.7, zeta_band=ZETA_BAND_TURN,
            feat_stop=0.0, feat_scale=180.0, feat_pwm=float(PWM_MAX),
            clamp_near_stop=False,
        )
        self.turn_pid   = NeuralPID()
        self._last_turn = 0.0

        # --- optional runtime logging of tuner outputs -----------------
        if DEBUG_LOG:
            try:
                log_dir = os.path.join(os.getcwd(), "Yuval")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, LOG_FILENAME)
                first = not os.path.exists(log_path)
                self._log_f = open(log_path, "a", newline="")
                self._log_writer = csv.writer(self._log_f)
                if first:
                    self._log_writer.writerow([
                        "ts", "state", "dist_front_m", "raw_speed", "ff", "speed",
                        "kp_d","ki_d","kd_d","zeta_d",
                        "kp_h","ki_h","kd_h","zeta_h",
                        "kp_t","ki_t","kd_t","zeta_t",
                        "correction","left","right",
                    ])
                    self._log_f.flush()
            except Exception as e:
                print("Warning: could not open tuner log:", e)
                self._log_f = None
                self._log_writer = None
        else:
            self._log_f = None
            self._log_writer = None

        self._stop_motors()
        print("Calibrating gyro (5 s) — DO NOT MOVE ROBOT...")

    # ── GYRO CALLBACK ─────────────────────────────────────────────────────
    def imu_cb(self, msg):
        raw = msg.vector.z
        now = time.monotonic()

        if self.prev_raw_yaw is None:
            self.prev_raw_yaw  = raw
            self.prev_raw_time = now
            return

        dt = now - self.prev_raw_time
        if dt <= 0:
            return

        rate = wrap_angle(raw - self.prev_raw_yaw) / dt
        self.prev_raw_yaw  = raw
        self.prev_raw_time = now

        # ── CALIBRATION PHASE ─────────────────────────────────────────────
        if not self.imu_ready:
            self.calib_rates.append(rate)
            if now - self.start_time >= IMU_CALIB_TIME:
                self.drift_offset = sum(self.calib_rates) / len(self.calib_rates)
                self.heading       = 0.0
                self.start_heading = 0.0
                self.imu_ready     = True
                self.state         = "FORWARD"
                print(f"Gyro calibrated  (drift offset = {self.drift_offset:.3f} deg/s)")
                print("Driving forward!")
            return

        # ── OPERATIONAL ───────────────────────────────────────────────────
        corrected = rate - self.drift_offset
        # Suppress small gyro noise only when we're intentionally holding a
        # heading. During a commanded turn (or any active rotation), slow
        # rotation is signal and must not be zeroed out.
        hold_heading_states = ("FORWARD", "WAIT", "DECIDE", "SETTLE", "INIT")
        if self.state in hold_heading_states and abs(corrected) < GYRO_DEADBAND:
            corrected = 0.0
        self.heading += corrected * dt

    # ── LIDAR CALLBACK (sensor mounted 180° flipped, three cones) ───────
    #   Front  → minimum of valid rays  (safety-critical: stop on any close ray)
    #   Left/Right → mean of valid rays (decision-making: average is noise-robust)
    def scan_cb(self, msg):
        # During a commanded rotation the lidar is not used by the turn loop
        # (turns are closed on the gyro heading only). Processing every ray here
        # blocks the single-threaded executor, delaying imu_cb and the turn
        # completion check, which makes the robot overshoot the target angle.
        # Skip the work while TURNING; SETTLE (robot stationary) refreshes the
        # forward distance again before the next FORWARD leg.
        if self.state == "TURN":
            return

        front_min  = float("inf")
        left_sum,  left_n  = 0.0, 0
        right_sum, right_n = 0.0, 0

        half_pi = math.pi / 2.0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):                   continue
            if r < msg.range_min or r > msg.range_max: continue
            if r > MAX_RANGE:                          continue

            angle = wrap_rad(msg.angle_min + i * msg.angle_increment + math.pi)

            if abs(angle) <= self.cone_front:
                if r < front_min:
                    front_min = r
            elif abs(angle - half_pi) <= self.cone_side:
                left_sum  += r;  left_n  += 1
            elif abs(angle + half_pi) <= self.cone_side:
                right_sum += r;  right_n += 1

        if front_min < float("inf"):
            self.dist_front = front_min
        if left_n > 0:
            self.dist_left  = left_sum  / left_n
        if right_n > 0:
            self.dist_right = right_sum / right_n


    # ── CONTROL LOOP (forward logic from working version) ─────────────────
    def control(self):
        if not self.imu_ready:
            return

        heading = self.heading

        # ── FORWARD: PID far away, fixed 30% near wall ───────────────────
        if self.state == "FORWARD":

            # wall arrival check (debounced)
            if self.dist_front <= STOP_DIST:
                self._stop_motors()
                self.obstacle_count += 1
                if self.obstacle_count >= CONFIRM_SCANS:
                    self.pause_until = time.monotonic() + WAIT_SECONDS
                    self.state = "WAIT"
                    print(f"Wall at {self.dist_front*100:.1f} cm — waiting {WAIT_SECONDS:.0f} s")
                return  # always stop driving once in the stop zone
            else:
                self.obstacle_count = 0

            # ── Speed: MLP tunes (kp,ki,kd); dist_pid integrates the error
            kp_d, ki_d, kd_d = self.dist_tuner.maybe_update(
                self.dist_front, self._last_speed, time.monotonic()
            )
            dist_err = self.dist_front - STOP_DIST
            raw_speed = self.dist_pid.step(dist_err, kp_d, ki_d, kd_d)

            # Feedforward from distance to boost mid-range speed (smoothly).
            norm = clamp((self.dist_front - STOP_DIST) / (MAX_RANGE - STOP_DIST), 0.0, 1.0)
            ff = MIN_PWM + (CRUISE_PWM - MIN_PWM) * (norm ** SPEED_FF_EXP)

            speed = SPEED_FF_WEIGHT * ff + (1.0 - SPEED_FF_WEIGHT) * raw_speed
            speed = clamp(speed, MIN_PWM, CRUISE_PWM)
            self._last_speed = speed

            # ── Heading: MLP tunes heading PID gains each loop
            head_err = self.start_heading - heading
            kp_h, ki_h, kd_h = self.head_tuner.maybe_update(
                head_err, self._last_corr, time.monotonic()
            )
            correction = self.head_pid.step(head_err, kp_h, ki_h, kd_h)
            self._last_corr = correction

            left  = clamp(speed + correction, 0, PWM_MAX)
            right = clamp(speed - correction, 0, PWM_MAX)
            print(f"[FWD] dist={self.dist_front*100:.0f}cm  "
                  f"kp={kp_d:.1f} ki={ki_d:.2f} kd={kd_d:.1f}  "
                  f"speed={speed:.0f}% (pid={raw_speed:.0f} ff={ff:.0f})  "
                  f"steering={correction:+.0f}  L={left:.0f}%  R={right:.0f}%")

            if DEBUG_LOG and getattr(self, "_log_writer", None):
                try:
                    zeta_d = kd_d / (2.0 * math.sqrt(max(kp_d + 0.1 * ki_d, 1e-9)))
                    zeta_h = kd_h / (2.0 * math.sqrt(max(kp_h + 0.1 * ki_h, 1e-9)))
                    if hasattr(self, "turn_tuner") and getattr(self.turn_tuner, "gains", None) is not None:
                        tg = self.turn_tuner.gains
                        kp_t = float(tg[0]); ki_t = float(tg[1]); kd_t = float(tg[2])
                        zeta_t = kd_t / (2.0 * math.sqrt(max(kp_t + 0.1 * ki_t, 1e-9)))
                    else:
                        kp_t = ki_t = kd_t = zeta_t = 0.0
                    ts = time.time()
                    self._log_writer.writerow([
                        ts, self.state, round(self.dist_front, 3), round(raw_speed, 3), round(ff, 3), round(speed, 3),
                        round(kp_d, 3), round(ki_d, 3), round(kd_d, 3), round(zeta_d, 3),
                        round(kp_h, 3), round(ki_h, 3), round(kd_h, 3), round(zeta_h, 3),
                        round(kp_t, 3), round(ki_t, 3), round(kd_t, 3), round(zeta_t, 3),
                        round(correction, 3), round(left, 3), round(right, 3),
                    ])
                    try:
                        self._log_f.flush()
                    except Exception:
                        pass
                except Exception:
                    pass

            self._drive_forward(left, right)
            return

        # ── WAIT: hold motors off, then read LiDAR and decide direction ──
        if self.state == "WAIT":
            self._stop_motors()
            if time.monotonic() >= self.pause_until:
                self.state = "DECIDE"
            return

        # ── DECIDE: choose turn direction from averaged LiDAR side readings ──
        if self.state == "DECIDE":
            # Fall back to open space if a cone had no valid rays this scan
            left  = self.dist_left  if math.isfinite(self.dist_left)  else float("inf")
            right = self.dist_right if math.isfinite(self.dist_right) else float("inf")
            diff  = left - right   # positive → more space on left

            print(f"  DECIDE  left={left*100:.0f}cm  right={right*100:.0f}cm  "
                  f"diff={diff*100:+.0f}cm  deadband=±{TURN_DEADBAND*100:.0f}cm")

            if left < DEADEND_SIDE_DIST and right < DEADEND_SIDE_DIST:
                # Both sides are closer than DEADEND_SIDE_DIST — this is a dead end.
                # A U-turn (179°) is the only escape.  179° is used instead of 180°
                # because wrap_angle(±180) flips sign and confuses the PID direction.
                delta = DEADEND_TURN_DEG
                print("  → Dead end (both sides blocked) — U-turn 180°")

            elif diff > TURN_DEADBAND:
                # Left average distance exceeds right by more than TURN_DEADBAND —
                # there is meaningfully more space on the left, so turn left (+90°).
                delta = TURN_ANGLE_DEG
                print(f"  → Left clearer by {diff*100:.0f}cm — turning left +90°")

            elif diff < -TURN_DEADBAND:
                # Right average distance exceeds left by more than TURN_DEADBAND —
                # there is meaningfully more space on the right, so turn right (−90°).
                delta = -TURN_ANGLE_DEG
                print(f"  → Right clearer by {abs(diff)*100:.0f}cm — turning right −90°")

            elif self.dist_front <= STOP_DIST:
                # Sides are nearly equal AND the front is still blocked — going
                # straight would immediately re-trigger the wall stop and loop
                # forever.  The only exit is a U-turn.
                delta = DEADEND_TURN_DEG
                print(f"  → Sides equal (diff {abs(diff)*100:.0f}cm) but front "
                      f"still blocked — U-turn 180°")

            else:
                # Sides are nearly equal and the front has cleared — safe to go
                # straight and re-evaluate at the next obstacle.
                self.start_heading  = heading
                self.obstacle_count = 0
                self.head_pid.reset()
                self.dist_pid.reset()
                self.dist_tuner.feat.reset()
                self._last_speed = float(MIN_PWM)
                self.state = "FORWARD"
                print(f"  → Sides nearly equal (diff {abs(diff)*100:.0f}cm < "
                      f"{TURN_DEADBAND*100:.0f}cm deadband) — going straight")
                return

            # Arm the NeuroPID turn
            self.turn_target     = heading + delta
            self.turn_started_at = time.monotonic()
            self.turn_pid.reset()
            self.state = "TURN"
            print(f"  → target heading = {self.turn_target:.1f}°")
            return

        # ── TURN: NeuroPID-controlled rotation (90° or 180°) ─────────────
        if self.state == "TURN":
            error = wrap_angle(self.turn_target - heading)

            # Completed: within 1° of target — enter settle pause
            if abs(error) < TURN_DONE_DEG:
                self._stop_motors()
                self.settle_until = time.monotonic() + SETTLE_SECONDS
                self.state = "SETTLE"
                print(f"Turn complete  (heading = {heading:.1f}°) — "
                      f"settling for {SETTLE_SECONDS:.1f} s")
                return

            # Safety timeout — enter settle anyway so robot stabilises
            if time.monotonic() - self.turn_started_at > TURN_TIMEOUT:
                self._stop_motors()
                self.settle_until = time.monotonic() + SETTLE_SECONDS
                self.state = "SETTLE"
                print(f"Turn timeout  (error = {error:.1f}°) — "
                      f"settling for {SETTLE_SECONDS:.1f} s")
                return

            # Scale the motor floor down as we close in on the target.
            # At ≥20° away: floor = TURN_MIN_PWM.  At <20°: floor ramps toward 10%
            # so the robot decelerates instead of crashing through the setpoint.
            turn_min = clamp(int(TURN_MIN_PWM * abs(error) / 20.0), 10, TURN_MIN_PWM)
            kp_t, ki_t, kd_t = self.turn_tuner.maybe_update(
                error, self._last_turn, time.monotonic()
            )
            cmd = self.turn_pid.step(error, kp_t, ki_t, kd_t)
            spd = clamp(abs(cmd), turn_min, PWM_MAX)
            # enforce absolute minimum turn speed
            spd = max(spd, TURN_MIN_PWM)
            self._last_turn = spd
            print(f"[TURN] err={error:+.1f}°  kp={kp_t:.1f} ki={ki_t:.2f} kd={kd_t:.1f} "
                  f"cmd={cmd:+.1f}%  spd={spd:.0f}%")
            if cmd > 0:
                self._turn_left(spd)
            else:
                self._turn_right(spd)
            return

        # ── SETTLE: hold motors off for SETTLE_SECONDS after a turn ──────
        if self.state == "SETTLE":
            self._stop_motors()
            if time.monotonic() >= self.settle_until:
                # Capture the actual heading after settling — this becomes
                # the new straight-ahead reference for the next forward leg.
                self.start_heading  = heading
                self.obstacle_count = 0
                self.head_pid.reset()
                self.dist_pid.reset()
                self.dist_tuner.feat.reset()
                self._last_speed = float(MIN_PWM)
                self.state = "FORWARD"
                print(f"Settled  (heading = {heading:.1f}°) — resuming forward")
            return

    # ── MOTOR HELPERS ─────────────────────────────────────────────────────
    def _drive_forward(self, l, r):
        gpio_set(IN1_LINE, 1); gpio_set(IN2_LINE, 0)
        gpio_set(IN3_LINE, 1); gpio_set(IN4_LINE, 0)
        self.left_pwm.set(l)
        self.right_pwm.set(r)

    def _turn_left(self, s):
        gpio_set(IN1_LINE, 1); gpio_set(IN2_LINE, 0)
        gpio_set(IN3_LINE, 0); gpio_set(IN4_LINE, 1)
        self.left_pwm.set(s)
        self.right_pwm.set(s)

    def _turn_right(self, s):
        gpio_set(IN1_LINE, 0); gpio_set(IN2_LINE, 1)
        gpio_set(IN3_LINE, 1); gpio_set(IN4_LINE, 0)
        self.left_pwm.set(s)
        self.right_pwm.set(s)

    def _stop_motors(self):
        self.left_pwm.set(0); self.right_pwm.set(0)
        for l in (IN1_LINE, IN2_LINE, IN3_LINE, IN4_LINE):
            gpio_set(l, 0)

    def stop(self):
        self._stop_motors()
        self.left_pwm.stop()
        self.right_pwm.stop()
        if hasattr(self, "_log_f") and self._log_f:
            try:
                self._log_f.close()
            except Exception:
                pass



# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    bot = Robot()
    try:
        while rclpy.ok():
            rclpy.spin_once(bot, timeout_sec=0.01)
            bot.control()
    except KeyboardInterrupt:
        pass
    finally:
        bot.stop()
        rclpy.shutdown()
        print("Shutdown complete")


if __name__ == "__main__":
    main()
