import Jetson.GPIO as GPIO
import smbus
import time
GPIO.setmode(GPIO.BOARD)
# MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
GYRO_CONFIG = 0x1B
GYRO_ZOUT_H = 0x47
# Initialize I2C (SMBus)
bus = smbus.SMBus(7)
device_address = 0x68
# Wake up the MPU6050 as it starts in sleep mode
bus.write_byte_data(device_address, PWR_MGMT_1, 0)
time.sleep(0.1)
bus.write_byte_data(device_address, GYRO_CONFIG, 0x00)  # ±250°/s
def read_gyro_z():
    """Read Z-axis gyroscope in degrees/second"""
    high = bus.read_byte_data(device_address, GYRO_ZOUT_H)
    low = bus.read_byte_data(device_address, GYRO_ZOUT_H + 1)
    value = (high << 8) + low
    if value >= 0x8000:
        value -= 0x10000
    return value / 131.0  # deg/s for ±250°/s range
# Calibration: calculate gyroscope offset
print("Calibrating gyroscope... Keep sensor still!")
gyro_offset = 0.0
calibration_samples = 1000
for i in range(calibration_samples):
    gyro_offset += read_gyro_z()
    if (i + 1) % 100 == 0:
        print(f"Calibration progress: {i + 1}/{calibration_samples}")
    time.sleep(0.001)
gyro_offset /= calibration_samples
print(f"Calibration complete! Offset: {gyro_offset:.4f}°/s\n")
# Variables for angle calculation
angle_z = 0.0
dt = 0.01  # sampling interval in seconds
last_time = time.time()
GYRO_THRESHOLD = 0.02  # Adjust this threshold based on your sensor noise
print("Reading angle... (Press Ctrl+C to stop)")
print("=" * 50)
try:
    while True:
        # Calculate actual time delta for more accuracy
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Read gyroscope and apply calibration
        gyro_z = read_gyro_z() - gyro_offset
        
        # Dead zone filter: suppress small readings (reduces drift)
        if abs(gyro_z) < GYRO_THRESHOLD:
            gyro_z = 0
        
        # Integrate to get angle
        angle_z += gyro_z * dt
        
        # Wrap angle to -180 to +180 degrees (more intuitive than 0-360)
        while angle_z > 180:
            angle_z -= 360
        while angle_z < -180:
            angle_z += 360
        
        # Display
        print(f"Gyro Z: {gyro_z:7.2f}°/s | Angle Z: {angle_z:7.2f}°")
        
        time.sleep(0.01)  # ~100Hz sampling rate
        
except KeyboardInterrupt:
    print("\n\nProgram stopped by user")
    GPIO.cleanup()
