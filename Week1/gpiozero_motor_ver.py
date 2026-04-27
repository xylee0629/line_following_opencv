from gpiozero import Motor
from time import sleep

#IN1, IN2, IN3, IN4, ENA, ENB are 19, 13, 6, 5, 26, 22 on BCM respectively 
motor1 = Motor(19, 13, enable = 26)
motor2 = Motor(6, 5, enable = 22)
speed = 0.5

# The default of the frequency for the Motor class is 100Hz.
# The parameter in the forward funtion, "speed" represents the duty cycle of the PWM. 

# Functions for movement
def forwardmotion():
    motor1.forward(speed)
    motor2.forward(speed)
    print("Forward")
    
def backwardmotion():
    motor1.backward(speed)
    motor2.backward(speed)
    print("Backwards")

# left wheel forward, turning right on the spot
def rightturn():
    motor1.forward(speed)
    motor2.backward(speed)
    print("Left")

# right wheel forward, turn left on the spot
def leftturn():
    motor1.backward(speed)
    motor2.forward(speed)
    print("Right")

# The main function     
try:
    forwardmotion()
    sleep(5)
    leftturn()
    sleep(5)
    backwardmotion()
    sleep(5)
    rightturn()
    sleep(5)
    
finally:
    # No need for GPIO cleanup as the library automatically resets the pins 
    motor1.stop()
    motor2.stop()