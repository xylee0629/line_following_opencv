import cv2
from picamera2 import Picamera2
import numpy as np
import time
from gpiozero import Motor, PWMOutputDevice, Servo
import math

# ----------------------------- 
# Camera parameters
# ----------------------------- 
frame_width = 640
frame_height = 360
frame_centre = frame_width / 2

# ----------------------------- 
# Motor & PID parameters
# ----------------------------- 
frequency = 600
dutyCycle = 0.5

# PID constants
Kp = 0.004
Kd = 0.004
Ki = 0.0
last_error = 0

# ----------------------------- 
# Robot motors
# ----------------------------- 
motorLeft = Motor(19, 13)
ENA = PWMOutputDevice(26, frequency=frequency)
motorRight = Motor(6, 5)
ENB = PWMOutputDevice(22, frequency=frequency)
servo = Servo(20)

# ----------------------------- 
# PID calculation
# ----------------------------- 
def calculatePID(cx):
    global last_error
    error = cx - frame_centre
    P = Kp * error
    D = Kd * (error - last_error)
    I = Ki * error
    last_error = error
    control = P + I + D

    left_pwm = dutyCycle + control
    right_pwm = dutyCycle - control
    left_pwm = max(-1.0, min(1.0, left_pwm))
    right_pwm = max(-1.0, min(1.0, right_pwm))
    return left_pwm, right_pwm

# ----------------------------- 
# Motor movement
# ----------------------------- 
def move(left_pwm, right_pwm):
    if left_pwm >= 0:
        motorLeft.forward()
        ENA.value = abs(left_pwm)
    else:
        motorLeft.backward()
        ENA.value = abs(left_pwm)
        
    if right_pwm >= 0:
        motorRight.forward()
        ENB.value = abs(right_pwm)
    else:
        motorRight.backward()
        ENB.value = abs(right_pwm)

def stopMotors():
    motorLeft.stop()
    motorRight.stop()
    time.sleep(0.5)

# ----------------------------- 
# Auto HSV Calibration Tool
# ----------------------------- 
def calibrate_hsv(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Manually set the initial color range for calibration
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([180, 255, 255])

    # Get the current ROI from the center of the frame for calibration
    roi = hsv[frame_height//4:3*frame_height//4, frame_width//4:3*frame_width//4]

    # Calculate histogram for the ROI (this will help us figure out the predominant color)
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])

    # Find the max value in the histogram to determine the color range
    max_val = np.argmax(hist)
    print(f"Predominant hue in the ROI: {max_val}")

    # Adjust the hue based on the predominant color
    # Here we are considering red, yellow, black based on the hue range of the histogram
    if 0 <= max_val < 10:  # Red
        lower_bound = np.array([0, 100, 100])
        upper_bound = np.array([10, 255, 255])
    elif 20 <= max_val < 40:  # Yellow
        lower_bound = np.array([20, 100, 100])
        upper_bound = np.array([30, 255, 255])
    elif 0 <= max_val <= 179:  # Black
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 50])

    return lower_bound, upper_bound

# ----------------------------- 
# Initialize Camera
# ----------------------------- 
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (frame_width, frame_height)}))
picam2.start()
time.sleep(1)

robot_active = False

# ----------------------------- 
# Main Loop - Robot Movement
# ----------------------------- 
try:
    while True:
        frame = picam2.capture_array()

        # Auto calibrate HSV range on a section of the frame
        lower_bound, upper_bound = calibrate_hsv(frame)

        # Create a mask based on the calibrated color range
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find contours of the masked image (get the largest contour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = frame_centre
        else:
            cx = frame_centre  # If no contours, assume the center

        # Calculate PID for motor control
        left_pwm, right_pwm = calculatePID(cx)

        if robot_active:
            move(left_pwm, right_pwm)
        else:
            move(0, 0)
            cv2.putText(frame, "Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw contours for visualization
        if largest_contour is not None:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

        # Show the live camera feed
        cv2.imshow("Robot Camera", frame)

        # ----------------------------- 
        # Keyboard input
        # ----------------------------- 
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            robot_active = True
            print("Motors started")

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    stopMotors()
    ENA.close()
    ENB.close()
    picam2.stop()
    cv2.destroyAllWindows()