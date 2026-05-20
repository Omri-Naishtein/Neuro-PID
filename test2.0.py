import math
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Vector3Stamped
from simple_pid import PID

# ── GPIO CONFIG ───────────────────────────────────────────────────────────────
CHIP     = "gpiochip0"
ENA_LINE = 41
ENB_LINE = 43
IN1_LINE = 53
IN2_LINE = 113
IN3_LINE = 52
IN4_LINE = 51

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PWM_MAX          = 60   # ⬅️ CHANGED (was 40)
DRIVE_PWM        = 45

STOP_DIST        = 0.25
CONFIRM_SCANS    = 3

WAIT_SECONDS     = 3.0
TURN_ANGLE_DEG   = 90.0
TURN_DONE_DEG    = 2.0
TURN_TIMEOUT     = 5.0

IMU_CALIB_TIME   = 5.0   # ⬅️ NEW

FRONT_CONE_DEG   = 20.0
MAX_RANGE        = 2.0

# ── PID ──────────────────────────────────────────────────────────────────────
DRIVE_KP, DRIVE_KI, DRIVE_KD = 1.8, 0.0, 0.10
TURN_KP, TURN_KI, TURN_KD    = 2.2, 0.0, 0.12

YAW_ALPHA = 0.15

# ── HELPERS ───────────────────────────────────────────────────────────────────
def gpio_set(line, value):
    subprocess.run(["gpioset", CHIP, f"{line}={value}"], check=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def wrap_angle(a):
    return (a + 180.0) % 360.0 - 180.0

def angle_diff(target, current):
    return wrap_angle(target - current)

# ── PWM ───────────────────────────────────────────────────────────────────────
class SoftPWM:
    def __init__(self, line, freq=100):
        self.line = line
        self.freq = freq
        self.duty = 0
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        period = 1.0 / self.freq
        while self.running:
            if self.duty <= 0:
                gpio_set(self.line, 0)
                time.sleep(period)
            elif self.duty >= 100:
                gpio_set(self.line, 1)
                time.sleep(period)
            else:
                gpio_set(self.line, 1)
                time.sleep(period * self.duty / 100)
                gpio_set(self.line, 0)
                time.sleep(period * (1 - self.duty / 100))

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
            depth=10
        )

        self.create_subscription(LaserScan, "/scan", self.scan_cb, qos)
        self.create_subscription(Vector3Stamped, "/imu/euler_angles", self.imu_cb, 10)

        self.left_pwm = SoftPWM(ENA_LINE)
        self.right_pwm = SoftPWM(ENB_LINE)

        # sensors
        self.latest_dist = float("inf")
        self.cone = math.radians(FRONT_CONE_DEG)

        self.yaw_filt = None
        self.imu_ready = False

        # state
        self.state = "INIT"
        self.obstacle_count = 0

        self.turn_target = 0
        self.drive_heading = 0
        self.turn_started_at = 0

        self.pause_until = 0

        # calibration
        self.start_time = time.monotonic()

        # PID
        self.pid_drive = PID(DRIVE_KP, DRIVE_KI, DRIVE_KD, setpoint=0)
        self.pid_drive.output_limits = (-10, 10)

        self.pid_turn = PID(TURN_KP, TURN_KI, TURN_KD, setpoint=0)
        self.pid_turn.output_limits = (-PWM_MAX, PWM_MAX)

        self._stop_motors()
        print("Calibrating IMU (5 seconds)...")

    # ── IMU ────────────────────────────────────────────────────────────────
    def imu_cb(self, msg):
        yaw = msg.vector.z

        if self.yaw_filt is None:
            self.yaw_filt = yaw
        else:
            self.yaw_filt = YAW_ALPHA * yaw + (1 - YAW_ALPHA) * self.yaw_filt

        # ⬅️ NEW: wait 5 seconds before allowing movement
        if not self.imu_ready:
            if time.monotonic() - self.start_time < IMU_CALIB_TIME:
                return

            self.imu_ready = True
            self.drive_heading = wrap_angle(self.yaw_filt)
            self.state = "FORWARD"
            print("IMU calibrated → driving forward")

    # ── LIDAR (UNCHANGED) ───────────────────────────────────────────────────
    def scan_cb(self, msg):
        min_d = float("inf")

        for i, r in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            angle = (angle + math.pi) % (2 * math.pi) - math.pi

            if (math.pi - abs(angle)) > self.cone:
                continue

            if math.isfinite(r) and r < msg.range_min and r > 0.0:
                min_d = 0.0
                break

            if not math.isfinite(r):
                continue
            if r < msg.range_min or r > msg.range_max:
                continue
            if r > MAX_RANGE:
                continue

            if r < min_d:
                min_d = r

        self.latest_dist = min_d

    # ── CONTROL LOOP ───────────────────────────────────────────────────────
    def control(self):
        if not self.imu_ready:
            return

        now = time.monotonic()
        yaw = wrap_angle(self.yaw_filt)

        # ── FORWARD ─────────────────────────────────────────────
        if self.state == "FORWARD":

            if self.latest_dist <= STOP_DIST:
                self.obstacle_count += 1
            else:
                self.obstacle_count = 0

            if self.obstacle_count >= CONFIRM_SCANS:
                self._stop_motors()

                print(f"\nSTOPPED at {self.latest_dist*100:.1f} cm")
                print("Waiting 3 seconds...")

                self.state = "WAIT"
                self.pause_until = now + WAIT_SECONDS
                return

            correction = self.pid_drive(yaw)

            left = clamp(DRIVE_PWM - correction, 0, PWM_MAX)
            right = clamp(DRIVE_PWM + correction, 0, PWM_MAX)

            self._drive_forward(left, right)
            return

        # ── WAIT ───────────────────────────────────────────────
        if self.state == "WAIT":
            self._stop_motors()

            if now >= self.pause_until:
                self.turn_target = wrap_angle(yaw + TURN_ANGLE_DEG)
                self.pid_turn.setpoint = self.turn_target
                self.pid_turn.reset()

                self.turn_started_at = now
                self.state = "TURN"

                print("Starting 90° turn...")
            return

        # ── TURN ───────────────────────────────────────────────
        if self.state == "TURN":

            error = angle_diff(self.turn_target, yaw)

            if abs(error) < TURN_DONE_DEG:
                self._stop_motors()

                self.drive_heading = wrap_angle(yaw)
                self.pid_drive.setpoint = self.drive_heading
                self.pid_drive.reset()

                self.obstacle_count = 0
                self.state = "FORWARD"

                print("TURN DONE → forward")
                return

            if now - self.turn_started_at > TURN_TIMEOUT:
                self._stop_motors()

                self.drive_heading = wrap_angle(yaw)
                self.pid_drive.setpoint = self.drive_heading
                self.pid_drive.reset()

                self.obstacle_count = 0
                self.state = "FORWARD"

                print("TURN TIMEOUT → forward")
                return

            turn_pwm = clamp(20 + abs(error) * 0.3, 15, PWM_MAX)

            if error > 0:
                self._turn_left(turn_pwm)
            else:
                self._turn_right(turn_pwm)

    # ── MOTOR CONTROL ───────────────────────────────────────────────
    def _drive_forward(self, l, r):
        gpio_set(IN1_LINE, 1)
        gpio_set(IN2_LINE, 0)
        gpio_set(IN3_LINE, 1)
        gpio_set(IN4_LINE, 0)

        self.left_pwm.set(l)
        self.right_pwm.set(r)

    def _turn_left(self, s):
        gpio_set(IN1_LINE, 1)
        gpio_set(IN2_LINE, 0)
        gpio_set(IN3_LINE, 0)
        gpio_set(IN4_LINE, 1)

        self.left_pwm.set(s)
        self.right_pwm.set(s)

    def _turn_right(self, s):
        gpio_set(IN1_LINE, 0)
        gpio_set(IN2_LINE, 1)
        gpio_set(IN3_LINE, 1)
        gpio_set(IN4_LINE, 0)

        self.left_pwm.set(s)
        self.right_pwm.set(s)

    def _stop_motors(self):
        self.left_pwm.set(0)
        self.right_pwm.set(0)

        for l in [IN1_LINE, IN2_LINE, IN3_LINE, IN4_LINE]:
            gpio_set(l, 0)

    def stop(self):
        self._stop_motors()
        self.left_pwm.stop()
        self.right_pwm.stop()

# ── MAIN ───────────────────────────────────────────────────────────────
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
