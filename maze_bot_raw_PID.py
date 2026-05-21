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

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PWM_MAX        = 75       # motor cap (%)
CRUISE_PWM     = 60       # max forward speed
MIN_PWM        = 25       # minimum motor speed — lower so it can creep near wall

STOP_DIST      = 0.3    # m — stop earlier to absorb momentum
CONFIRM_SCANS  = 2        # react faster to wall
WAIT_SECONDS   = 2.0      # pause before turning

TURN_ANGLE_DEG = 90.0
TURN_DONE_DEG  = 2.0
TURN_TIMEOUT   = 6.0

IMU_CALIB_TIME = 5.0      # seconds to calibrate gyro offset

FRONT_CONE_DEG = 20.0
MAX_RANGE      = 2.0

BRAKE_ZONE     = 0.40     # metres before wall where speed scaling begins

# deadband for gyro rate (deg/s)
GYRO_DEADBAND  = 1.0

# ── PID GAINS ─────────────────────────────────────────────────────────────────
HEAD_KP, HEAD_KI, HEAD_KD = 2.0,  0.15, 0.05   # heading correction
DIST_KP, DIST_KI, DIST_KD = 30.0, 0.5,  8.0    # distance → speed (Kd brakes early)
TURN_KP, TURN_KI, TURN_KD = 2.0,  0.3,  0.7    # 90° rotation


# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpio_set(line, value):
    subprocess.run(["gpioset", CHIP, f"{line}={value}"], check=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    """Wrap to [-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0


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

        # lidar
        self.latest_dist = float("inf")
        self.cone = math.radians(FRONT_CONE_DEG)

        # state machine: INIT → FORWARD → WAIT → TURN → FORWARD (circular)
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

    # ── LIDAR CALLBACK ────────────────────────────────────────────────────
    # lidar is mounted backwards — front of robot = back of lidar (±π)
    def scan_cb(self, msg):
        min_d = float("inf")
        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            if (math.pi - abs(angle)) > self.cone:
                continue
            if math.isfinite(r) and r > 0.0 and r < msg.range_min:
                min_d = 0.0
                break
            if not math.isfinite(r):                   continue
            if r < msg.range_min or r > msg.range_max: continue
            if r > MAX_RANGE:                          continue
            if r < min_d:
                min_d = r
        self.latest_dist = min_d

    # ── CONTROL LOOP ──────────────────────────────────────────────────────
    def control(self):
        if not self.imu_ready:
            return

        heading = self.heading

        # ── FORWARD: drive straight + slow down near wall ─────────────────
        if self.state == "FORWARD":

            # wall arrival check (debounced)
            if self.latest_dist <= STOP_DIST:
                self.obstacle_count += 1
            else:
                self.obstacle_count = 0

            if self.obstacle_count >= CONFIRM_SCANS:
                self._stop_motors()
                self.pause_until = time.monotonic() + WAIT_SECONDS
                self.state = "WAIT"
                print(f"Wall at {self.latest_dist*100:.1f} cm — waiting {WAIT_SECONDS:.0f} s")
                return

            # ── Speed PID with brake-zone scaling ─────────────────────────
            # In the last BRAKE_ZONE metres the effective speed cap shrinks
            # linearly so the robot arrives slow regardless of PID output.
            dist_err = self.latest_dist - STOP_DIST

            if self.latest_dist < BRAKE_ZONE:
                scale        = self.latest_dist / BRAKE_ZONE   # 0.0 → 1.0
                effective_max = max(MIN_PWM, CRUISE_PWM * scale)
            else:
                effective_max = CRUISE_PWM

            speed = self.pid_dist.step(dist_err)
            speed = clamp(speed, 0, effective_max)             # respect scaled cap

            # Enforce MIN_PWM only when NOT dangerously close to the wall
            if 0 < speed < MIN_PWM and self.latest_dist > STOP_DIST + 0.05:
                speed = MIN_PWM

            # ── Heading PID keeps the robot driving straight ───────────────
            head_err   = self.start_heading - heading
            correction = self.pid_head.step(head_err)

            # ┌──────────────────────────────────────────────────────────────┐
            # │  If the robot STILL circles, swap the + and − below:        │
            # │     left  = clamp(speed - correction, 0, PWM_MAX)           │
            # │     right = clamp(speed + correction, 0, PWM_MAX)           │
            # └──────────────────────────────────────────────────────────────┘
            left  = clamp(speed + correction, 0, PWM_MAX)
            right = clamp(speed - correction, 0, PWM_MAX)
            self._drive_forward(left, right)
            return

        # ── WAIT: stop for WAIT_SECONDS ───────────────────────────────────
        if self.state == "WAIT":
            self._stop_motors()
            if time.monotonic() >= self.pause_until:
                self.turn_target     = heading + TURN_ANGLE_DEG
                self.turn_started_at = time.monotonic()
                self.pid_turn.reset()
                self.state = "TURN"
                print("Starting 90° turn")
            return

        # ── TURN: rotate 90° in place ─────────────────────────────────────
        if self.state == "TURN":
            error = self.turn_target - heading

            if abs(error) < TURN_DONE_DEG:
                self._stop_motors()
                self.start_heading  = heading   # new reference for next straight leg
                self.obstacle_count = 0
                self.pid_head.reset()           # clear heading integrator for new leg
                self.pid_dist.reset()           # clear distance integrator for new leg
                self.state = "FORWARD"
                print(f"Turn complete  (heading = {heading:.1f}°) → forward again")
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
