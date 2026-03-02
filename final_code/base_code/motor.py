# General motor code that is used as a base for any movement of the car. 
""" To do: 
    Change dutyCycle control from variable that is inputted at the start and fixed for the code to a general variable that can be tweaked when needed.
    Update both foward.py and turn.py to use the same name for the same variable."""

from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice
import time
import math

# Variables for motor control
frequency = 500
dutyCycle = 0.7

# Variables for encoder
encoderState = {
    "left": 0,
    "right": 0
}
diameter = 8.5
encoder_slots = 20

# Variables used for function turn_degrees
track_width = 20.5  

# Equations for calculation
circumference = math.pi * diameter
distance_pulse = circumference / encoder_slots

# Motor GPIO pins
motor1 = Motor(19, 13)
ENA = PWMOutputDevice(26,frequency=frequency)
motor2 = Motor(6, 5)
ENB = PWMOutputDevice(22,frequency=frequency)

# Encoder GPIO pins 
encoder_left = DigitalInputDevice(27,pull_up=False)
encoder_right = DigitalInputDevice(21,pull_up=False)



# Function for encoder 
def updateEncoder(side):
    encoderState[side] += 1

def updateDistance():
    global distance_left, distance_right, distance_total
    distance_left = (encoderState["left"] * circumference) / encoder_slots
    distance_right = (encoderState["right"] * circumference) / encoder_slots
    distance_total = (distance_left + distance_right) / 2
    
# Functions for movement
def forward_motion():
    motor1.forward()
    ENA.value = dutyCycle
    motor2.forward()
    ENB.value = dutyCycle
    
def backward_motion():
    motor1.backward()
    ENA.value = dutyCycle
    motor2.backward()
    ENB.value = dutyCycle
    
def stop_motor():
    motor1.stop()
    motor2.stop()
    time.sleep(0.5)
    
# Turns the robot specific degrees. Clockwise (Right) = Positive angle; Anticlockwise (Left) = Negative Angle 
def turn_degrees(angle):
    
    # Reset counters to 0 so we can count just for this turn
    encoderState["left"] = 0
    encoderState["right"] = 0
    
    # Calculate the target arc length
    # Formula: (Angle / 360) * (Pi * Track_Width)
    target_distance = (abs(angle) / 360) * (math.pi * track_width)
    
    # Calculate how many pulses equal that distance
    target_pulses = target_distance / distance_pulse
    
    print(f"Target Arc: {target_distance:.2f}cm | Target Pulses: {int(target_pulses)}")

    # Set motor direction
    if angle > 0:
        # Turn Right: Left Motor Forward, Right Motor Backward
        motor1.forward()
        ENA.value = dutyCycle
        motor2.backward()
        ENB.value = dutyCycle
    else:
        # Turn Left: Left Motor Backward, Right Motor Forward
        motor1.backward()
        ENA.value = dutyCycle
        motor2.forward()
        ENB.value = dutyCycle

    # Blocking Loop: Wait until we reach the target pulses
    while True:
        # We average the pulses from both sides for better accuracy
        global avg_pulses
        avg_pulses = (encoderState["left"] + encoderState["right"] ) / 2
        
        if avg_pulses >= target_pulses:
            break
        
        # Small sleep to prevent CPU hogging
        time.sleep(0.001)

    # Stop
    stop_motor()
    print(f"Turn Complete. Pulses: {avg_pulses}. Distance: {avg_pulses * distance_pulse:.2f}")
    
    
# Track encoder pulses    
encoder_left.when_activated = lambda: updateEncoder("left")
encoder_right.when_activated = lambda: updateEncoder("right")

try:
    # Move forward for 10s
    updateDistance()
    forward_motion()
    time.sleep(10)
    
finally:
    stop_motor()
    
# Old functions 
"""def straightMotion(distance_cm):
    resetEncoder()
    target_ticks = calculateTicks(abs(distance_cm))
    
    if distance_cm > 0:
        motorLeft.forward()
        ENA.value = dutyCycle
        motorRight.forward()
        ENB.value = dutyCycle
    elif distance_cm < 0:
        motorLeft.backward()
        ENA.value = dutyCycle
        motorRight.backward()
        ENB.value = dutyCycle
    else :
        return
    
    while True:
        avg_pulses = (encoderState["left"] + encoderState["right"]) / 2
        if avg_pulses >= target_ticks:
            break
        
    stopMotors()
    return avg_pulses

def turnDegrees(angle):
    resetEncoder()
    
    target_distance_arc = (abs(angle) / 360) * (math.pi * track_width)
    target_ticks = calculateTicks(target_distance_arc)

    if angle > 0:
        motorLeft.forward()
        ENA.value = dutyCycle
        motorRight.forward()
        ENB.value = dutyCycle
    elif angle < 0:
        motorLeft.backward()
        ENA.value = dutyCycle
        motorRight.backward()
        ENB.value = dutyCycle
    else:
        return
    
    while True:
        avg_pulses = (encoderState["left"] + encoderState["right"]) / 2
        if avg_pulses >= target_ticks:
            break
    stopMotors()
    return avg_pulses"""