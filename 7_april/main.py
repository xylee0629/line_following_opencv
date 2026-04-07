# main.py
import cv2
import time
import threading
from picamera2 import Picamera2
import numpy as np

import config
from motor_control import MotorController
from vision_utils import VisionAnalyzer
from cli_manager import TerminalController

# ==========================================
# SHARED RESOURCES
# ==========================================
class SharedFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def write(self, frame):
        with self.lock:
            self.frame = frame 

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.left_pwm = 0.0
        self.right_pwm = 0.0
        self.line_draw_data = None # Holds the contour for the UI to draw

    def update_steering(self, left, right, draw_data):
        with self.lock:
            self.left_pwm = left
            self.right_pwm = right
            self.line_draw_data = draw_data

    def get_steering(self):
        with self.lock:
            return self.left_pwm, self.right_pwm

    def get_draw_data(self):
        with self.lock:
            return self.line_draw_data

# ==========================================
# THREAD WORKER FUNCTIONS
# ==========================================
def line_follower_thread(vision, motors, frame_buffer, shared_state, cli):
    left_flag, right_flag = 0, 0
    
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            bottom_roi = frame[240:480, :]
            
            # Unpack the 4 variables
            cx, left_flag, right_flag, draw_data = vision.process_line(bottom_roi, left_flag, right_flag)
            left_pwm, right_pwm = motors.calculate_pid(cx)
            
            # Pass steering AND draw data to shared state
            shared_state.update_steering(left_pwm, right_pwm, draw_data)
            
        time.sleep(0.01) 

def motor_control_thread(motors, shared_state, cli):
    while cli.app_running:
        left_pwm, right_pwm = shared_state.get_steering()
        
        if cli.robot_active:
            motors.move(left_pwm, right_pwm)
        else:
            motors.stop()
            
        time.sleep(0.01) 

# ==========================================
# MAIN INITIALIZATION & CAMERA LOOP
# ==========================================
def main():
    motors = MotorController()
    vision = VisionAnalyzer()
    cli = TerminalController()
    
    frame_buffer = SharedFrameBuffer()
    shared_state = SharedState()

    cli.start()

    t1 = threading.Thread(target=line_follower_thread, args=(vision, motors, frame_buffer, shared_state, cli), daemon=True)
    t2 = threading.Thread(target=motor_control_thread, args=(motors, shared_state, cli), daemon=True)
    
    t1.start()
    t2.start()

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format": 'BGR888', "size": (config.FRAME_WIDTH, config.FRAME_HEIGHT)}))
    picam2.start()
    time.sleep(1) 

    try:
        print("[SYSTEM] Line Follower Architecture Running. Feed is active.")
        
        # Main Camera Loop
        while cli.app_running:
            # 1. Grab the raw frame and fix the colors
            raw_frame = picam2.capture_array()
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
            # 2. Write the untouched, raw frame to the buffer for the Vision Thread
            frame_buffer.write(raw_frame)

            # 3. THE FIX: Create a separate copy purely for drawing UI
            display_frame = raw_frame.copy()

            # ==========================================
            # UI DRAWING LOGIC (Draw ONLY on display_frame)
            # ==========================================
            draw_data = shared_state.get_draw_data()
            
            # Draw boundary line
            cv2.line(display_frame, (0, 240), (config.FRAME_WIDTH, 240), (255, 255, 255), 1)

            if draw_data is not None:
                contour, color, cx, cy = draw_data
                
                # Slice the display frame, not the raw frame!
                bottom_display_roi = display_frame[240:480, :]
                
                # Draw the shape of the line
                cv2.drawContours(bottom_display_roi, [contour], -1, color, 3)
                
                # Draw the magenta dot
                cv2.circle(bottom_display_roi, (cx, cy), 6, (255, 0, 255), -1)
                
                # Draw the tracking line
                center_x = int(config.FRAME_CENTRE)
                cv2.line(bottom_display_roi, (center_x, cy), (cx, cy), (255, 255, 255), 2)

            # Display Feed
            cv2.imshow("Robot View", display_frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\nProgram stopped by Ctrl+C") 
    finally:
        cli.app_running = False
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        motors.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()