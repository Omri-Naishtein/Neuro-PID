import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped

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
CRUISE_PWM     = 60       # max forward speed
MIN_PWM        = 25       # minimum motor speed — lower so it can creep near wall

STOP_DIST      = 0.20     # m — full stop
CREEP_DIST     = 0.40     # m — below this, fixed 30% power
CREEP_PWM      = 30       # fixed motor power in creep zone
CONFIRM_SCANS  = 2        # require 2 scans to confirm wall
WAIT_SECONDS   = 2.0      # pause before turning

TURN_ANGLE_DEG = 90.0
TURN_DONE_DEG  = 2.0
TURN_TIMEOUT   = 6.0

IMU_CALIB_TIME = 5.0      # seconds to calibrate gyro offset

# ── LIDAR CONE WIDTHS (degrees half-angle each side) ──────────────────────────
FRONT_CONE_DEG = 5.0      # ±5° narrow cone ahead
SIDE_CONE_DEG  = 5.0      # ±5° cone for left/right sensing
MAX_RANGE      = 2.0      # ignore returns beyond this (m)

# deadband for gyro rate (deg/s)
GYRO_DEADBAND  = 1.0

# ── PID GAINS (from working version) ─────────────────────────────────────────
HEAD_KP, HEAD_KI, HEAD_KD = 2.0,  0.15, 0.05   # heading correction
DIST_KP, DIST_KI, DIST_KD = 80.0, 0.5,  8.0    # distance → speed
TURN_KP, TURN_KI, TURN_KD = 2.0,  0.3,  0.7    # 90° rotation


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


# ── PID ───────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.prev_time  = None

    def step(self, error):
        now = time.monotonic()
        dt  = 0.02 if self.prev_time is None else max(1e-3, now - self.prev_time)
        self.prev_time = now

        self.integral  += error * dt
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error

        out = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.out_min is not None: out = max(self.out_min, out)
        if self.out_max is not None: out = min(self.out_max, out)
        return out


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
        # INIT -> FORWARD -> WAIT -> DECIDE -> TURN -> FORWARD (repeat)
        self.state           = "INIT"
        self.start_heading   = 0.0
        self.turn_target     = 0.0
        self.turn_started_at = 0.0
        self.pause_until     = 0.0
        self.obstacle_count  = 0
        self.start_time      = time.monotonic()

        # PID controllers
        self.pid_head = PID(HEAD_KP, HEAD_KI, HEAD_KD, out_min=-20,      out_max=20)
        self.pid_dist = PID(DIST_KP, DIST_KI, DIST_KD, out_min=0,        out_max=CRUISE_PWM)
        self.pid_turn = PID(TURN_KP, TURN_KI, TURN_KD, out_min=-PWM_MAX, out_max=PWM_MAX)

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
        if abs(corrected) < GYRO_DEADBAND:
            corrected = 0.0
        self.heading += corrected * dt

    # ── LIDAR CALLBACK (three ±5° cones, sensor mounted 180° flipped) ────
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

        print(f"  Intersection scan -> front={f*100:.0f}cm  "
              f"left={l*100:.0f}cm  right={r*100:.0f}cm")

        ALL_BLOCKED_CM = 0.30

        all_blocked = (f < ALL_BLOCKED_CM and
                       l < ALL_BLOCKED_CM and
                       r < ALL_BLOCKED_CM)

        if all_blocked:
            print("  Decision: TURN BACK (180)")
            return 180.0

        if f >= l and f >= r:
            print("  Decision: GO STRAIGHT")
            return 0.0

        if l >= r:
            print("  Decision: TURN LEFT (+90)")
            return 90.0

        print("  Decision: TURN RIGHT (-90)")
        return -90.0

    # ── CONTROL LOOP (forward logic from working version) ─────────────────
    def control(self):
        if not self.imu_ready:
            return

        heading = self.heading

        # ── FORWARD: PID far away, fixed 30% near wall ───────────────────
        if self.state == "FORWARD":

            # wall arrival check (debounced)
            if self.dist_front <= STOP_DIST:
                self.obstacle_count += 1
            else:
                self.obstacle_count = 0

            if self.obstacle_count >= CONFIRM_SCANS:
                self._stop_motors()
                self.pause_until = time.monotonic() + WAIT_SECONDS
                self.state = "WAIT"
                print(f"Wall at {self.dist_front*100:.1f} cm — waiting {WAIT_SECONDS:.0f} s")
                return

            # ── Speed: PID when far, fixed creep when close ───────────────
            if self.dist_front > CREEP_DIST:
                # normal PID
                dist_err = self.dist_front - STOP_DIST
                speed = self.pid_dist.step(dist_err)
                speed = clamp(speed, MIN_PWM, CRUISE_PWM)
            else:
                # creep zone (40 cm → 20 cm): fixed 30% power
                self.pid_dist.reset()
                speed = CREEP_PWM

            # ── Heading PID keeps the robot driving straight ──────────────
            head_err   = self.start_heading - heading
            correction = self.pid_head.step(head_err)

            left  = clamp(speed + correction, 0, PWM_MAX)
            right = clamp(speed - correction, 0, PWM_MAX)
            zone = "CREEP" if self.dist_front <= CREEP_DIST else "PID"
            print(f"[{zone}] dist={self.dist_front*100:.0f}cm  speed={speed:.0f}%  "
                  f"steering={correction:+.0f}  L={left:.0f}%  R={right:.0f}%")
            self._drive_forward(left, right)
            return

        # ── WAIT: stop for WAIT_SECONDS, then decide ──────────────────────
        if self.state == "WAIT":
            self._stop_motors()
            if time.monotonic() >= self.pause_until:
                self.state = "DECIDE"
            return

        # ── DECIDE: read all three cones, pick the best direction ─────────
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
                print(f"Turning {delta:+.0f}  (target heading = {self.turn_target:.1f})")
            return

        # ── TURN: rotate to target heading ────────────────────────────────
        if self.state == "TURN":
            error = wrap_angle(self.turn_target - heading)

            if abs(error) < TURN_DONE_DEG:
                self._stop_motors()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head.reset()
                self.pid_dist.reset()
                self.state = "FORWARD"
                print(f"Turn complete  (heading = {heading:.1f}) -> forward")
                return

            if time.monotonic() - self.turn_started_at > TURN_TIMEOUT:
                self._stop_motors()
                self.start_heading  = heading
                self.obstacle_count = 0
                self.pid_head.reset()
                self.pid_dist.reset()
                self.state = "FORWARD"
                print(f"Turn timeout  (error = {error:.1f}) -> forward anyway")
                return

            cmd = self.pid_turn.step(error)
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
