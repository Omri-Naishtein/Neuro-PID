import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped

from neuro_pid import PID as NeuroPID, angle_diff

# ── GPIO CONFIG ───────────────────────────────────────────────────────────────
CHIP     = "gpiochip0"
ENA_LINE = 41
ENB_LINE = 43
IN1_LINE = 53
IN2_LINE = 113
IN3_LINE = 52
IN4_LINE = 51

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PWM_MAX        = 75       # motor cap (%)
CRUISE_PWM     = 60       # max forward speed
MIN_PWM        = 50       # minimum motor speed
CREEP_PWM      = 20       # fixed slow speed in creep zone (50→20 cm)

STOP_DIST      = 0.20     # m — stop distance from wall
CREEP_DIST     = 0.50     # m — below this distance, disable speed PID and creep
CONFIRM_SCANS  = 1        # stop immediately when wall is detected
WAIT_SECONDS   = 2.0      # pause at intersection before deciding

TURN_DONE_DEG  = 2.0
TURN_TIMEOUT   = 6.0

IMU_CALIB_TIME = 5.0      # seconds to calibrate gyro offset

# ── LIDAR CONE WIDTHS (degrees half-angle each side) ──────────────────────────
FRONT_CONE_DEG = 20.0
SIDE_CONE_DEG  = 20.0
MAX_RANGE      = 2.0      # ignore returns beyond this (m)

# deadband for gyro rate (deg/s)
GYRO_DEADBAND  = 1.0

# ── NEURO-PID PHYSICS PARAMETERS ─────────────────────────────────────────────
# Robot mass: 1.87 kg  |  Damping: ζ = 0.9 (well-damped)
# Note: ω₀ differs per controller (heading needs gentle gains, turn needs aggressive)

# Heading correction (keeps the robot driving straight)
# ω₀=5 made Kp≈55 → saturated on tiny errors, integrator wound up.
# Lower ω₀ gives a softer, more stable correction.
HEAD_ZETA    = 0.9
HEAD_OMEGA   = 2.0        # ↓ from 5: smooth steering, no saturation
HEAD_MASS    = 1.87
HEAD_OUT_MAX = 20.0       # max steering correction (%)

# Distance → speed (how aggressively the robot brakes for walls)
DIST_ZETA    = 0.9
DIST_OMEGA   = 5.0
DIST_MASS    = 1.87

# Turn (90° / 180° rotation in place)
TURN_ZETA    = 0.9
TURN_OMEGA   = 5.0
TURN_MASS    = 1.87

# ── DIRECTION FLIPS (try these if motors are wired backwards) ────────────────
# Your log shows the robot turned the WRONG WAY during -90° (rotated +130°).
# If forward driving still circles or turns go the wrong direction, flip these.
FWD_CORR_DIR = +1         # flip to -1 if forward driving still curves
TURN_CMD_DIR = +1         # flip to -1 if 90° turns go the wrong way


# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpio_set(line, value):
    subprocess.run(["gpioset", CHIP, f"{line}={value}"], check=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_rad(a):
    """Wrap angle to [-π, π] radians."""
    return (a + math.pi) % (2 * math.pi) - math.pi


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

        # ── lidar distances (updated every scan) ──────────────────────────
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

        # ── neuro-PID controllers (created fresh on each state transition) ─
        self.pid_head = None
        self.pid_dist = None
        self.pid_turn = None

        self._stop_motors()
        print("Calibrating gyro (5 s) — DO NOT MOVE ROBOT...")

    # ── PID FACTORY METHODS ───────────────────────────────────────────────
    # Creating a fresh NeuroPID each time a state is entered replaces the
    # old .reset() calls and guarantees zeroed integrators / filter state.

    def _new_head_pid(self):
        return NeuroPID(
            setpoint=0.0,
            zeta=HEAD_ZETA, omega_n=HEAD_OMEGA, mass=HEAD_MASS,
            output_max=HEAD_OUT_MAX, output_min=0.0,
            done_threshold=0.1,   # never "done" — runs continuously
            feedback=False, graph=False,
        )

    def _new_dist_pid(self):
        return NeuroPID(
            setpoint=0.0,
            zeta=DIST_ZETA, omega_n=DIST_OMEGA, mass=DIST_MASS,
            output_max=CRUISE_PWM, output_min=0.0,
            done_threshold=0.005, # never "done" while driving
            feedback=False, graph=False,
        )

    def _new_turn_pid(self):
        return NeuroPID(
            setpoint=0.0,
            zeta=TURN_ZETA, omega_n=TURN_OMEGA, mass=TURN_MASS,
            output_max=PWM_MAX, output_min=MIN_PWM,
            done_threshold=TURN_DONE_DEG,
            feedback=False, graph=False,
        )

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

        rate = angle_diff(raw, self.prev_raw_yaw) / dt
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
                self.pid_head = self._new_head_pid()
                self.pid_dist = self._new_dist_pid()
                print(f"Gyro calibrated  (drift offset = {self.drift_offset:.3f} deg/s)")
                print("Driving forward!")
            return

        # ── OPERATIONAL ───────────────────────────────────────────────────
        corrected = rate - self.drift_offset
        if abs(corrected) < GYRO_DEADBAND:
            corrected = 0.0
        self.heading += corrected * dt

    # ── LIDAR CALLBACK ────────────────────────────────────────────────────
    def scan_cb(self, msg):
        front_min = float("inf")
        left_min  = float("inf")
        right_min = float("inf")

        half_pi = math.pi / 2.0

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r):                    continue
            if r < msg.range_min or r > msg.range_max:  continue
            if r > MAX_RANGE:                           continue

            raw_angle = msg.angle_min + i * msg.angle_increment
            angle     = wrap_rad(raw_angle + math.pi)

            if abs(angle) <= self.cone_front:
                if r < front_min:
                    front_min = r
            elif abs(angle - half_pi) <= self.cone_side:
                if r < left_min:
                    left_min = r
            elif abs(angle + half_pi) <= self.cone_side:
                if r < right_min:
                    right_min = r

        if front_min < float("inf"):
            self.dist_front = front_min
        if left_min < float("inf"):
            self.dist_left  = left_min
        if right_min < float("inf"):
            self.dist_right = right_min

    # ── INTERSECTION DECISION ─────────────────────────────────────────────
    def _decide_direction(self):
        f = self.dist_front
        l = self.dist_left
        r = self.dist_right

        print(f"  Intersection scan → front={f*100:.0f}cm  "
              f"left={l*100:.0f}cm  right={r*100:.0f}cm")

        ALL_BLOCKED_CM = 0.30

        all_blocked = (f < ALL_BLOCKED_CM and
                       l < ALL_BLOCKED_CM and
                       r < ALL_BLOCKED_CM)

        if all_blocked:
            print("  Decision: TURN BACK (180°)")
            return 180.0

        if f >= l and f >= r:
            print("  Decision: GO STRAIGHT")
            return 0.0

        if l >= r:
            print("  Decision: TURN LEFT (+90°)")
            return 90.0

        print("  Decision: TURN RIGHT (-90°)")
        return -90.0

    # ── CONTROL LOOP ──────────────────────────────────────────────────────
    def control(self):
        if not self.imu_ready:
            return

        heading = self.heading

        # ── FORWARD ───────────────────────────────────────────────────────
        if self.state == "FORWARD":

            if self.dist_front <= STOP_DIST:
                self.obstacle_count += 1
            else:
                self.obstacle_count = 0

            if self.obstacle_count >= CONFIRM_SCANS:
                self._stop_motors()
                self.pause_until = time.monotonic() + WAIT_SECONDS
                self.state = "WAIT"
                print(f"Wall ahead at {self.dist_front*100:.1f} cm — pausing {WAIT_SECONDS:.0f} s")
                return

            # ── speed from distance PID (or creep) ───────────────────────
            if self.dist_front > CREEP_DIST:
                dist_err = self.dist_front - STOP_DIST
                if dist_err < 0.0:
                    dist_err = 0.0
                raw_speed = self.pid_dist(dist_err)
                speed = clamp(abs(raw_speed), 0, CRUISE_PWM)
            else:
                # creep zone — fixed slow speed, fresh PID for when we leave
                self.pid_dist = self._new_dist_pid()
                speed = CREEP_PWM

            # ── heading correction via neuro_pid ─────────────────────────
            head_err   = angle_diff(self.start_heading, heading)
            correction = self.pid_head(head_err)

            # cap correction to 30 % of current speed
            max_correction = speed * 0.30
            correction = clamp(correction, -max_correction, max_correction)

            left  = clamp(speed + correction, 0, PWM_MAX)
            right = clamp(speed - correction, 0, PWM_MAX)
            print(f"[FORWARD] front={self.dist_front*100:.1f}cm  "
                  f"speed={speed:.1f}  correction={correction:+.1f}  "
                  f"L={left:.1f}%  R={right:.1f}%")
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
                self.pid_head = self._new_head_pid()
                self.pid_dist = self._new_dist_pid()
                self.state = "FORWARD"
                print("Going straight through intersection")
            else:
                self.turn_target     = heading + delta
                self.turn_started_at = time.monotonic()
                self.pid_turn = self._new_turn_pid()
                self.state = "TURN"
                print(f"Turning {delta:+.0f}°  (target heading = {self.turn_target:.1f}°)")
            return

        # ── TURN ──────────────────────────────────────────────────────────
        if self.state == "TURN":
            error = angle_diff(self.turn_target, heading)

            # timeout guard
            if time.monotonic() - self.turn_started_at > TURN_TIMEOUT:
                self._stop_motors()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head = self._new_head_pid()
                self.pid_dist = self._new_dist_pid()
                self.state = "FORWARD"
                print(f"Turn timeout  (error = {error:.1f}°) → forward anyway")
                return

            cmd = self.pid_turn(error)

            # neuro_pid sets .done when |error| < done_threshold
            if self.pid_turn.done:
                self._stop_motors()
                self.pid_turn.finish()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head = self._new_head_pid()
                self.pid_dist = self._new_dist_pid()
                self.state = "FORWARD"
                print(f"Turn complete  (heading = {heading:.1f}°) → forward")
                return

            spd = clamp(abs(cmd), MIN_PWM, PWM_MAX)
            if cmd > 0:
                self._turn_left(spd)
            else:
                self._turn_right(spd)
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
