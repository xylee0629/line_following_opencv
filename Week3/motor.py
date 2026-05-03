from gpiozero import Motor, PWMOutputDevice
import config as config

# 1. Declare variables globally, but DO NOT initialize the hardware yet!
_motors_ready = False
motorLeft = None
motorRight = None
ENA = None
ENB = None

def _setup():
    """Hidden function to initialize hardware ONLY when a motor command is actually sent."""
    global _motors_ready, motorLeft, motorRight, ENA, ENB
    if not _motors_ready:
        motorLeft = Motor(19, 13)
        ENA = PWMOutputDevice(26, frequency=config.FREQUENCY)
        motorRight = Motor(6, 5)
        ENB = PWMOutputDevice(22, frequency=config.FREQUENCY)
        _motors_ready = True

def move(left_pwm, right_pwm):
    # 2. Trigger setup on the first move command 
    _setup() 
    
    left_value = max(-1.0, min(1.0, left_pwm / 100.0))
    right_value = max(-1.0, min(1.0, right_pwm / 100.0))
    
    if left_value >= 0:
        motorLeft.forward()
    else:
        motorLeft.backward()
    ENA.value = abs(left_value)
    
    if right_value >= 0:
        motorRight.forward()
    else:
        motorRight.backward()
    ENB.value = abs(right_value)
    
def stop():
    _setup()
    motorLeft.stop()
    motorRight.stop()
    ENA.value = 0
    ENB.value = 0
    
def cleanup():
    # Only try to clean up if the motors were actually initialized
    if _motors_ready:
        stop()
