"""
Maze robot — drive straight, turn 90° at walls.

State machine  (from working version):
  INIT → FORWARD → WAIT → DECIDE → TURN → FORWARD (repeat)

Three NeuroPID controllers:
  1. pid_dist    — forward speed from dist_front feedback.
                   error = dist_front − STOP_DIST
                   Naturally full-speed when far, slows to zero at STOP_DIST.
  2. pid_heading — keeps straight line while driving (gyro feedback).
  3. pid_turn    — executes 90° / 180° turns (gyro feedback).

Weights are saved on shutdown and reloaded on next boot (NPD4 format).
"""

import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped

from neuropid import PID

# ── GPIO CONFIG ───────────────────────────────────────────────────────────────
CHIP     = "gpiochip0"
ENA_LINE = 41
ENB_LINE = 43
IN1_LINE = 53
IN2_LINE = 113
IN3_LINE = 52
IN4_LINE = 51

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PWM_MAX       = 75        # motor cap (%)
CRUISE_PWM    = 60        # drive PID output ceiling
MIN_PWM       = 25        # minimum motor speed

STOP_DIST     = 0.20      # m — target stopping distance (PID setpoint)
CONFIRM_SCANS = 2         # consecutive scans required to confirm wall
WAIT_SECONDS  = 2.0       # pause before deciding turn direction

TURN_DONE_DEG = 2.0       # degrees — turn complete tolerance
TURN_TIMEOUT  = 6.0       # seconds — give up on turn

IMU_CALIB_TIME = 5.0      # seconds for gyro offset calibration

FRONT_CONE_DEG = 5.0      # ±deg cone ahead
SIDE_CONE_DEG  = 5.0      # ±deg cone left / right
MAX_RANGE      = 2.0      # ignore lidar returns beyond this (m)

GYRO_DEADBAND  = 1.0      # deg/s — ignore gyro noise below this

WEIGHTS_FILE   = "~/pid_weights.npd4"

# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpio_set(line, value):
    subprocess.run(["gpioset", CHIP, f"{line}={value}"], check=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0

def wrap_rad(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def fmt(d):
    return f"{d*100:.0f}cm" if math.isfinite(d) else "---"


# ── SOFTWARE PWM ──────────────────────────────────────────────────────────────
class SoftPWM:
    def __init__(self, line, freq=100):
        self.line  = line
        self.freq  = freq
        self.duty  = 0
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        period = 1.0 / self.freq
        while self.running:
            d = self.duty
            if d <= 0:
                gpio_set(self.line, 0); time.sleep(period)
            elif d >= 100:
                gpio_set(self.line, 1); time.sleep(period)
            else:
                gpio_set(self.line, 1); time.sleep(period * d / 100)
                gpio_set(self.line, 0); time.sleep(period * (1 - d / 100))

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

        # ── lidar ─────────────────────────────────────────────────────────
        self.dist_front = float("inf")
        self.dist_left  = float("inf")
        self.dist_right = float("inf")
        self.cone_front = math.radians(FRONT_CONE_DEG)
        self.cone_side  = math.radians(SIDE_CONE_DEG)

        # ── state machine ─────────────────────────────────────────────────
        self.state           = "INIT"
        self.start_heading   = 0.0
        self.turn_target     = 0.0
        self.turn_started_at = 0.0
        self.pause_until     = 0.0
        self.obstacle_count  = 0
        self.start_time      = time.monotonic()

        # ── NeuroPID controllers ──────────────────────────────────────────

        # Distance PID — error = dist_front − STOP_DIST (metres)
        # Large positive error (far) → full CRUISE_PWM
        # Error → 0 (at STOP_DIST)   → output → 0  (natural stop)
        self.pid_dist = PID(
            kp=80.0,
            ki=0.5,
            kd=8.0,
            profile="low",
            out_min=0.0,
            out_max=float(CRUISE_PWM),
            gain_limits={
                "kp": (20.0, 200.0),
                "ki": (0.0,  10.0),
                "kd": (0.0,  40.0),
            },
            integral_limit=15.0,
            gain_alpha=0.05,
            default_dt=0.02,
            max_output=1.0,
            lr=0.003,
        )

        # Heading PID — error = degrees off course, output = PWM differential
        self.pid_head = PID(
            kp=2.0,
            ki=0.15,
            kd=0.05,
            profile="low",
            out_min=-20.0,
            out_max=20.0,
            gain_limits={
                "kp": (0.5, 10.0),
                "ki": (0.0,  2.0),
                "kd": (0.0,  2.0),
            },
            integral_limit=20.0,
            gain_alpha=0.05,
            default_dt=0.02,
            max_output=1.0,
            lr=0.003,
        )

        # Turn PID — error = degrees remaining, output = turn motor speed
        self.pid_turn = PID(
            kp=2.0,
            ki=0.3,
            kd=0.7,
            profile="low",
            out_min=-float(PWM_MAX),
            out_max=float(PWM_MAX),
            gain_limits={
                "kp": (0.5, 15.0),
                "ki": (0.0,  5.0),
                "kd": (0.0, 10.0),
            },
            integral_limit=30.0,
            gain_alpha=0.08,
            default_dt=0.02,
            max_output=1.0,
            lr=0.005,
        )

        # Load weights from previous run (if any)
        import os
        path = os.path.expanduser(WEIGHTS_FILE)
        for name, pid in [("dist", self.pid_dist),
                          ("head", self.pid_head),
                          ("turn", self.pid_turn)]:
            fpath = path.replace(".npd4", f"_{name}.npd4")
            try:
                pid.load(fpath)
                g = pid.gains()
                print(f"  Loaded {name} weights  "
                      f"(Kp={g.kp:.2f} Ki={g.ki:.3f} Kd={g.kd:.3f})")
            except (OSError, ValueError):
                print(f"  No saved {name} weights — starting fresh")

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

        if not self.imu_ready:
            self.calib_rates.append(rate)
            if now - self.start_time >= IMU_CALIB_TIME:
                self.drift_offset = sum(self.calib_rates) / len(self.calib_rates)
                self.heading       = 0.0
                self.start_heading = 0.0
                self.imu_ready     = True
                self.state         = "FORWARD"
                print(f"Gyro calibrated  (drift = {self.drift_offset:.3f} deg/s)")
                print("Driving forward!")
            return

        corrected = rate - self.drift_offset
        if abs(corrected) < GYRO_DEADBAND:
            corrected = 0.0
        self.heading += corrected * dt

    # ── LIDAR CALLBACK ────────────────────────────────────────────────────
    def scan_cb(self, msg):
        front_min = float("inf")
        left_min  = float("inf")
        right_min = float("inf")
        half_pi   = math.pi / 2.0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):                   continue
            if r < msg.range_min or r > msg.range_max: continue
            if r > MAX_RANGE:                          continue

            angle = wrap_rad(msg.angle_min + i * msg.angle_increment + math.pi)

            if   abs(angle)             <= self.cone_front: front_min = min(front_min, r)
            elif abs(angle - half_pi)   <= self.cone_side:  left_min  = min(left_min,  r)
            elif abs(angle + half_pi)   <= self.cone_side:  right_min = min(right_min, r)

        if math.isfinite(front_min): self.dist_front = front_min
        if math.isfinite(left_min):  self.dist_left  = left_min
        if math.isfinite(right_min): self.dist_right = right_min

    # ── DIRECTION DECISION (from working version) ─────────────────────────
    def _decide_direction(self):
        f, l, r = self.dist_front, self.dist_left, self.dist_right
        print(f"  Scan → front={fmt(f)}  left={fmt(l)}  right={fmt(r)}")

        if f < 0.30 and l < 0.30 and r < 0.30:
            print("  Decision: TURN BACK (180°)")
            return 180.0

        if f >= l and f >= r:
            print("  Decision: GO STRAIGHT")
            return 0.0

        if l >= r:
            print("  Decision: TURN LEFT (+90°)")
            return 90.0

        print("  Decision: TURN RIGHT (−90°)")
        return -90.0

    # ── CONTROL LOOP ──────────────────────────────────────────────────────
    def control(self):
        if not self.imu_ready:
            return

        heading = self.heading

        # ── FORWARD ───────────────────────────────────────────────────────
        if self.state == "FORWARD":

            # Wall arrival check (debounced with CONFIRM_SCANS)
            if self.dist_front <= STOP_DIST:
                self.obstacle_count += 1
            else:
                self.obstacle_count = 0

            if self.obstacle_count >= CONFIRM_SCANS:
                self._stop_motors()
                self.pid_dist.reset()
                self.pause_until = time.monotonic() + WAIT_SECONDS
                self.state = "WAIT"
                print(f"Wall at {fmt(self.dist_front)} — waiting {WAIT_SECONDS:.0f} s")
                return

            # Distance PID: setpoint=STOP_DIST, measurement=dist_front
            # Caps dist to MAX_RANGE so inf never spikes the integrator.
            capped   = min(self.dist_front, MAX_RANGE)
            dist_err = capped - STOP_DIST
            speed    = self.pid_dist.step(dist_err, setpoint=STOP_DIST)
            speed    = clamp(speed, MIN_PWM, CRUISE_PWM)

            # Heading PID keeps robot straight
            head_err   = wrap_angle(self.start_heading - heading)
            correction = self.pid_head.step(head_err, setpoint=0.0)

            left  = clamp(speed + correction, 0, PWM_MAX)
            right = clamp(speed - correction, 0, PWM_MAX)

            gd = self.pid_dist.gains()
            print(f"[FWD] dist={fmt(self.dist_front)}  spd={speed:.0f}%  "
                  f"steer={correction:+.1f}  L={left:.0f}%  R={right:.0f}%  "
                  f"Kp={gd.kp:.1f}")
            self._drive_forward(left, right)
            return

        # ── WAIT ──────────────────────────────────────────────────────────
        if self.state == "WAIT":
            self._stop_motors()
            if time.monotonic() >= self.pause_until:
                self.state = "DECIDE"
            return

        # ── DECIDE ────────────────────────────────────────────────────────
        if self.state == "DECIDE":
            delta = self._decide_direction()

            if delta == 0.0:
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head.reset()
                self.pid_dist.reset()
                self.state = "FORWARD"
                print("Going straight through intersection")
            else:
                self.turn_target     = heading + delta
                self.turn_started_at = time.monotonic()
                self.pid_turn.reset()
                self.state = "TURN"
                print(f"Turning {delta:+.0f}°  (target = {self.turn_target:.1f}°)")
            return

        # ── TURN ──────────────────────────────────────────────────────────
        if self.state == "TURN":
            error = wrap_angle(self.turn_target - heading)

            if abs(error) < TURN_DONE_DEG:
                self._stop_motors()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head.reset()
                self.pid_dist.reset()
                self.state = "FORWARD"
                print(f"Turn complete  (heading = {heading:.1f}°) → forward")
                return

            if time.monotonic() - self.turn_started_at > TURN_TIMEOUT:
                self._stop_motors()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head.reset()
                self.pid_dist.reset()
                self.state = "FORWARD"
                print(f"Turn timeout  (error = {error:.1f}°) → forward anyway")
                return

            cmd = self.pid_turn.step(error, setpoint=self.turn_target)
            spd = clamp(abs(cmd), MIN_PWM, PWM_MAX)
            if cmd > 0:
                self._turn_left(spd)
            else:
                self._turn_right(spd)

            g = self.pid_turn.gains()
            print(f"[TURN] err={error:+.1f}°  spd={spd:.0f}%  "
                  f"Kp={g.kp:.2f} Ki={g.ki:.3f} Kd={g.kd:.3f}")
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
        self.left_pwm.set(0)
        self.right_pwm.set(0)
        for l in (IN1_LINE, IN2_LINE, IN3_LINE, IN4_LINE):
            gpio_set(l, 0)

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    def stop(self):
        self._stop_motors()
        self.left_pwm.stop()
        self.right_pwm.stop()

        import os
        path = os.path.expanduser(WEIGHTS_FILE)
        for name, pid in [("dist", self.pid_dist),
                          ("head", self.pid_head),
                          ("turn", self.pid_turn)]:
            fpath = path.replace(".npd4", f"_{name}.npd4")
            try:
                pid.save(fpath)
                g = pid.gains()
                print(f"  Saved {name} → {fpath}  "
                      f"(steps={pid.step_count}, "
                      f"Kp={g.kp:.2f} Ki={g.ki:.3f} Kd={g.kd:.3f})")
            except OSError as e:
                print(f"  Could not save {name}: {e}")


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
