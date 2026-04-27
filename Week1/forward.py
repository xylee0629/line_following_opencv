from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice
import time
import math

# The default of the frequency for the Motor class is 100Hz.
dutyCycle = 0.5
frequency = 500
diameter = 8.5
circumference = math.pi * diameter
count_per_pulse = 20
distance_target = 50.0
distance_total = 0

encoderState = {
    "left": 0,
    "right": 0
}

#IN1, IN2, IN3, IN4, ENA, ENB are 19, 13, 6, 5, 26, 22 on BCM respectively 
motor1 = Motor(19, 13)
ENA = PWMOutputDevice(26,frequency=frequency)
motor2 = Motor(6, 5)
ENB = PWMOutputDevice(22,frequency=frequency)

# Left and right Encoder
encoder_left = DigitalInputDevice(27,pull_up=False)
encoder_right = DigitalInputDevice(21,pull_up=False)

# Encoder functions 
def updateEncoder(side):
    encoderState[side] += 1

def updateDistance():
    global distance_left, distance_right, distance_total
    distance_left = (encoderState["left"] * circumference) / count_per_pulse
    distance_right = (encoderState["right"] * circumference) / count_per_pulse
    distance_total = (distance_left + distance_right) / 2
    
# Functions for movement
def forwardmotion():
    motor1.forward()
    ENA.value = dutyCycle
    motor2.forward()
    ENB.value = dutyCycle
    
def backwardmotion():
    motor1.backward()
    ENA.value = dutyCycle
    motor2.backward()
    ENB.value = dutyCycle
    
encoder_left.when_activated = lambda: updateEncoder("left")
encoder_right.when_activated = lambda: updateEncoder("right")

# The main function     
try:
    time_start = time.time()
    while distance_total < distance_target:
        updateDistance()
        print(f"left: {distance_left:.2f}, right: {distance_right:.2f}, average: {distance_total:.2f}")
        forwardmotion()
    print(f"Time taken: {time.time() - time_start:.2f}, Speed: {distance_total / (time.time() - time_start):.2f}")
    
finally:
    # No need for GPIO cleanup as the library automatically resets the pins 
    motor1.stop()
    motor2.stop()