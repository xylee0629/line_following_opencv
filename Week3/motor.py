#motor.py
from gpiozero import Motor, PWMOutputDevice
import config as config

class MotorController:
    def __init__(self):
        self.motorLeft = Motor(19, 13)
        self.ENA = PWMOutputDevice(26,frequency=config.FREQUENCY)
        self.motorRight = Motor(6, 5)
        self.ENB = PWMOutputDevice(22,frequency=config.FREQUENCY)
        
    def move(self, left_pwm, right_pwm):
        # Convert values frrom in terms of 100 to 0.1
        left_value = max(-1.0, min(1.0, left_pwm / 100.0))
        right_value = max(-1.0, min(1.0, right_pwm / 100.0))
        
        if left_value >= 0:
            self.motorLeft.forward()
        else:
            self.motorLeft.backward()
        self.ENA.value = abs(left_value)
        
        if right_value >= 0:
            self.motorRight.forward()
        else:
            self.motorRight.backward()
        self.ENB.value = abs(right_value)
        
    def stop(self):
        self.motorLeft.stop()
        self.motorRight.stop()
        self.ENA.value = 0
        self.ENB.value = 0
        
    def cleanup(self):
        self.stop()
        self.ENA.close()
        self.ENB.close()