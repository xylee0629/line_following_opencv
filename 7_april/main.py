# main.py
import cv2
import time
import threading
from picamera2 import Picamera2

import config
from motor_control import MotorController
from vision_utils import VisionAnalyzer
from cli_manager import TerminalController
from streamer import WebStreamer

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
        # Line Follower State
        self.left_pwm = 0.0
        self.right_pwm = 0.0
        
        # Symbol Detection State
        self.action_symbol = None

    # --- Line Follower Methods ---
    def update_steering(self, left, right):
        with self.lock:
            self.left_pwm = left
            self.right_pwm = right

    def get_steering(self):
        with self.lock:
            return self.left_pwm, self.right_pwm

    # --- Symbol Methods ---
    def update_symbol(self, symbol):
        with self.lock:
            self.action_symbol = symbol

    def consume_action_symbol(self):
        with self.lock:
            sym = self.action_symbol
            self.action_symbol = None # Motor clears it so it only reacts once
            return sym


# ==========================================
# THREAD WORKER FUNCTIONS
# ==========================================
def line_follower_thread(vision, motors, frame_buffer, shared_state, cli):
    left_flag, right_flag = 0, 0
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            bottom_roi = frame[240:480, :]
            
            # Unpack the 4 variables, but ignore the draw_data (using '_')
            cx, left_flag, right_flag, _ = vision.process_line(bottom_roi, left_flag, right_flag)
            left_pwm, right_pwm = motors.calculate_pid(cx)
            
            shared_state.update_steering(left_pwm, right_pwm)
            
        time.sleep(0.01) 


def symbol_detector_thread(vision, frame_buffer, shared_state, cli):
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            top_roi = frame[0:400, :]
            
            # Unpack the 2 variables, but ignore the bounding box (using '_')
            symbol, _ = vision.detect_symbol(top_roi)
            
            if symbol:
                shared_state.update_symbol(symbol)
                # Pause detection briefly to prevent spamming
                time.sleep(0.5)
                
        time.sleep(0.05) 


def motor_control_thread(motors, shared_state, cli):
    while cli.app_running:
        left_pwm, right_pwm = shared_state.get_steering()
        current_symbol = shared_state.consume_action_symbol()
        
        if cli.robot_active:
            # 1. Actionable Symbols override the line follower
            if current_symbol:
                print(f"\n[ACTION] Reacting to: {current_symbol}")
                
                if current_symbol == "Hazard":
                    motors.stop()
                    cli.robot_active = False
                    print("[STATE] Robot STOPPED. Type 's' + Enter to restart.")
                elif current_symbol == "Button":
                    motors.stop()
                    print("[ACTION] Pausing for 2 seconds...")
                    time.sleep(2.0)
                elif str(current_symbol).startswith("ARROW_"):
                    print(f"[ACTION] Navigating: {current_symbol}")
                    
            # 2. Drive normally
            else:
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
    
    streamer = WebStreamer(frame_buffer)
    streamer.start(port=5000)

    # Start CLI Listener (Terminal Input)
    cli.start()

    t1 = threading.Thread(target=line_follower_thread, args=(vision, motors, frame_buffer, shared_state, cli), daemon=True)
    t2 = threading.Thread(target=symbol_detector_thread, args=(vision, frame_buffer, shared_state, cli), daemon=True)
    t3 = threading.Thread(target=motor_control_thread, args=(motors, shared_state, cli), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format": 'BGR888', "size": (config.FRAME_WIDTH, config.FRAME_HEIGHT)}))
    picam2.start()
    time.sleep(1) 

    try:
        print("[SYSTEM] Headless Architecture Running. Robot is fully autonomous.")
        
        # The Main Loop now ONLY acts as a high-speed camera driver
        while cli.app_running:
            # 1. Grab raw frame and fix color
            raw_frame = picam2.capture_array()
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
            # 2. Send clean frame to the background vision threads
            frame_buffer.write(raw_frame)
            
            # Small sleep to prevent CPU pegging if capture_array returns instantly
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nProgram stopped by Ctrl+C") 
    finally:
        cli.app_running = False
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)
        t3.join(timeout=1.0)
        motors.cleanup()
        picam2.stop()
        # No cv2 windows to destroy anymore!

if __name__ == "__main__":
    main()