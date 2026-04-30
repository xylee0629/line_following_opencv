from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice
from signal import pause
from time import sleep
import math

frequency = 600
dutyCycle = 0.6
# --- HARDWARE SETUP ---
# IN1, IN2, IN3, IN4, ENA, ENB are 19, 13, 6, 5, 26, 22 on BCM respectively 
motor1 = Motor(19, 13)
ENA = PWMOutputDevice(26,frequency=frequency)
motor2 = Motor(6, 5)
ENB = PWMOutputDevice(22,frequency=frequency)
encoderleft = DigitalInputDevice(27, pull_up=False) # Left Encoder
encoderright = DigitalInputDevice(21, pull_up=False) # Right Encoder

# --- VARIABLES ---
diameter = 8.5
encoder_slots = 20
track_width = 20.5  # <--- Critical for turning accuracy

# --- CALCULATIONS ---
circumference = math.pi * diameter
distance_pulse = circumference / encoder_slots

encoder_data = {
    "left": 0,
    "right": 0
}

# --- ENCODER LOGIC ---
def updateEncoder(side):
    encoder_data[side] += 1
    
encoderleft.when_activated = lambda: updateEncoder("left")
encoderright.when_activated = lambda: updateEncoder("right")

# --- MOVEMENT FUNCTIONS ---

def stop_motors():
    motor1.stop()
    motor2.stop()
    sleep(0.5) # Short pause to let momentum settle

def turn_degrees(angle):
    """
    Turns the robot specific degrees.
    Positive angle = Turn Right
    Negative angle = Turn Left
    """
    print(f"\n--- Starting Turn: {angle} Degrees ---")
    
    # 1. Reset counters to 0 so we can count just for this turn
    encoder_data["left"] = 0
    encoder_data["right"] = 0
    
    # 2. Calculate the target arc length
    # Formula: (Angle / 360) * (Pi * Track_Width)
    target_distance = (abs(angle) / 360) * (math.pi * track_width)
    
    # 3. Calculate how many pulses equal that distance
    target_pulses = target_distance / distance_pulse
    
    print(f"Target Arc: {target_distance:.2f}cm | Target Pulses: {int(target_pulses)}")

    # 4. Set Motor Direction
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

    # 5. Blocking Loop: Wait until we reach the target pulses
    while True:
        # We average the pulses from both sides for better accuracy
        avg_pulses = (encoder_data["left"] + encoder_data["right"] ) / 2
        
        if avg_pulses >= target_pulses:
            break
        
        # Small sleep to prevent CPU hogging
        sleep(0.001)

    # 6. Stop
    stop_motors()
    print(f"Turn Complete. Pulses: {avg_pulses}. Distance: {avg_pulses * distance_pulse}")

# --- MAIN EXECUTION ---
try:
    print("System Ready.")
    
    turn_degrees(-180)
    

finally:
    motor1.stop()
    motor2.stop()
    print("Motors Released.")