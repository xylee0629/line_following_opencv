# PID tuned: dutyCycle 0.5, Kp = 0.003, Kd = 0.012, Ki = 0

import cv2
from picamera2 import Picamera2
import numpy as np
import time
from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice, Servo
import math

# Camera variables
frame_width = 640
frame_height = 360
frame_centre = frame_width / 2

# Motor variables 
frequency = 600
dutyCycle = 0.5
left_pwm = 0
right_pwm = 0

# Encoder variables
diameter = 8.5
encoder_slots = 20

# PID variables
Kp = 0.003
Kd = 0.013
Ki = 0.0
last_error = 0

# Calculations
circumference = math.pi * diameter
distance_per_tick = circumference / encoder_slots

encoderState = {
    "left": 0,
    "right": 0
}

# GPIO pins 
motorLeft = Motor(19, 13)
ENA = PWMOutputDevice(26,frequency=frequency)
motorRight = Motor(6, 5)
ENB = PWMOutputDevice(22,frequency=frequency)
encoderLeft = DigitalInputDevice(27, pull_up=False) 
encoderRight = DigitalInputDevice(21, pull_up=False) 
servo = Servo(20)

# Encoder functions
def updateEncoder(side):
    encoderState[side] += 1
    
def resetEncoder():
    encoderState["left"] = 0
    encoderState["right"] = 0

def calculateTicks(target_distance):
    return target_distance / distance_per_tick
    
# Movement functions 
def stopMotors():
    motorLeft.stop()
    motorRight.stop()
    time.sleep(0.5)

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
    
# PID calculation function
def calculatePID(cx, error, dutyCycle):
    integral = 0
    global last_error
    
    # Calculate error
    error = cx - frame_centre
    
    # PID calculation
    P = Kp * error
    integral += error
    I = Ki * integral
    derivative = error - last_error
    D = Kd * derivative
    control = P + I + D
    last_error = error
    
    
    # Apply to motors
    left_pwm = dutyCycle + control
    right_pwm = dutyCycle - control
    
    left_pwm = max(-1.0, min(1.0, left_pwm))
    right_pwm = max(-1.0, min(1.0, right_pwm))
    
    return left_pwm, right_pwm
    

# Initialise camera in video 
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (frame_width, frame_height)}))
picam2.start()

time.sleep(1)

left_flag = 0
right_flag = 0
robot_active = False

try:
    while True:
        #servo.min()

        frame = picam2.capture_array()
        
        # Converts frame to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blurs the grayscaled image (research the parameters)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Inverts the black line to white and find the threshold
        ret, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
        # Finds contours 
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If there is contours in the frame 
        if len(contours) > 0:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Get the moments (what is moments?)
            M = cv2.moments(largest_contour)
            
            if (M["m00"] != 0):
                # divide to find centre coordinates 
                cx = int(M["m10"] / M["m00"]) 
                cy = int(M["m01"] / M["m00"]) 
                
                # Update memory flags based on line position
                if cx <= frame_centre:
                    left_flag = 1
                    right_flag = 0
                elif cx > frame_centre:
                    left_flag = 0
                    right_flag = 1
            else:
                continue
        else:
            if left_flag == 1:
                cx = 0              # Set to most left
            elif right_flag == 1:
                cx = frame_width    # Set to most right
            else:
                cx = frame_centre   # Failsafe if lost on very first frame
                
        error = cx - frame_centre
        
        if error == 0:
            left_pwm = dutyCycle
            right_pwm = dutyCycle
        else: 
            left_pwm, right_pwm = calculatePID(cx, error, dutyCycle)
            
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        if robot_active:
            print(f"Target Speeds -> Left: {left_pwm:.2f} | Right: {right_pwm:.2f}")
            move(left_pwm, right_pwm)
        else:
            move(0, 0)
            cv2.putText(frame, "STANDBY - Press 's' to start", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        cv2.imshow("Original Frame", frame)
        # cv2.imshow("Threshold (Line is White)", thresh)
       
            
        # If keyboard presses q, quit the program
        key = cv2.waitKey(1) & 0xFF	
        if key == ord("q"):
            break
        elif key == ord("s"):
            robot_active = True
            print("Motors start")
    

except KeyboardInterrupt:
    print("Program stopped by user")   

finally:
    stopMotors()
    ENA.close()
    ENB.close()
    picam2.stop()
    cv2.destroyAllWindows()

