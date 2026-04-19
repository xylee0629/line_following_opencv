from gpiozero import Motor, PWMOutputDevice
import config

class MotorController:
    def __init__(self):
        self.motorLeft = Motor(19, 13)
        self.ENA = PWMOutputDevice(26, frequency=config.FREQUENCY)
        self.motorRight = Motor(6, 5)
        self.ENB = PWMOutputDevice(22, frequency=config.FREQUENCY)
        self.last_error = 0
        self.current_base_speed = config.DUTY_CYCLE 
        
    def set_profile(self, freq, base_speed):
        """Dynamically changes hardware frequency and base duty cycle."""
        self.ENA.frequency = freq 
        self.ENB.frequency = freq
        self.current_base_speed = base_speed

    def calculate_pid(self, cx):
        error = cx - config.FRAME_CENTRE
        P = config.KP * error
        D = config.KD * (error - self.last_error)
        control = P + D 
        self.last_error = error
        
        left_pwm = max(-1.0, min(1.0, self.current_base_speed + control))
        right_pwm = max(-1.0, min(1.0, self.current_base_speed - control))
        return left_pwm, right_pwm
        
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