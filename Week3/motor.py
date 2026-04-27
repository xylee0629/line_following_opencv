#motor.py
from gpiozero import Motor, PWMOutputDevice
import Week3.config as config

class MotorController:
    def __init__(self):
        self.motorLeft = Motor(19, 13)
        self.ENA = PWMOutputDevice(26,frequency=config.FREQUENCY)
        self.motorRight = Motor(6, 5)
        self.ENB = PWMOutputDevice(22,frequency=config.FREQUENCY)
        
    def move(self, left_pwm, right_pwm):
        if left_pwm >= 0:
            self.motorLeft.forward()
        else:
            self.motorLeft.backward()
        self.ENA.value = abs(left_pwm)
        
        if right_pwm >= 0:
            self.motorRight.forward()
        else:
            self.motorRight.backward()
        self.ENB.value = abs(right_pwm)
        
    def stop(self):
        self.motorLeft.stop()
        self.motorRight.stop()
        self.ENA.value = 0
        self.ENB.value = 0
        
    def cleanup(self):
        self.stop()
        self.ENA.close()
        self.ENB.close()