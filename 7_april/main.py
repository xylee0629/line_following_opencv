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
        self.line_draw_data = None
        
        # Symbol Detection State
        self.action_symbol = None

        self.ui_symbol_name = None
        self.ui_symbol_box = None
        self.ui_symbol_timer = 0
        
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

    def update_symbol(self, symbol, box=None):
        with self.lock:
            if symbol: # Only update if a valid symbol was found
                self.action_symbol = symbol
                self.ui_symbol_name = symbol
                self.ui_symbol_box = box
                self.ui_symbol_timer = time.time() # Start the 1-second clock

    def get_ui_symbol(self):
        with self.lock:
            # Only return the box if we saw the symbol less than 1 second ago
            if time.time() - self.ui_symbol_timer < 1.0:
                return self.ui_symbol_name, self.ui_symbol_box
            return None, None

    def consume_action_symbol(self):
        with self.lock:
            sym = self.action_symbol
            self.action_symbol = None 
            return sym


# ==========================================
# THREAD WORKER FUNCTIONS
# ==========================================
def line_follower_thread(vision, motors, frame_buffer, shared_state, cli):
    left_flag, right_flag = 0, 0
    
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            bottom_roi = frame[320:480, :]
            
            # Process line using priority logic (Red -> Yellow -> Black)
            cx, left_flag, right_flag, draw_data = vision.process_line(bottom_roi, left_flag, right_flag)
            
            # Calculate PID and update steering
            left_pwm, right_pwm = motors.calculate_pid(cx)
            shared_state.update_steering(left_pwm, right_pwm, draw_data)
            
        time.sleep(0.01)

def symbol_detector_thread(vision, frame_buffer, shared_state, cli):
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            top_roi = frame[0:400, :]
            
            # Catch both variables 
            symbol, box = vision.detect_symbol(top_roi)
            
            if symbol:
                shared_state.update_symbol(symbol, box)
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
                    direction = str(current_symbol).replace("ARROW_", "")
                    print(f"[ACTION] Blind turn sequence initiated: {direction}")
                    
                    # A. Move forward slightly to center over the intersection
                    motors.move(config.DUTY_CYCLE, config.DUTY_CYCLE)
                    time.sleep(0.3) # <-- ADJUST this time based on robot speed
                    
                    # B. Execute the blind turn
                    if direction == "LEFT":
                        motors.move(-config.DUTY_CYCLE, config.DUTY_CYCLE)
                    elif direction == "RIGHT":
                        motors.move(config.DUTY_CYCLE, -config.DUTY_CYCLE)
                        
                    time.sleep(0.4) # <-- ADJUST this time to achieve a ~90 degree turn
                    
                    # C. Stop briefly to stabilize
                    motors.stop()
                    time.sleep(0.1)
                    print("[ACTION] Turn complete. Snapping back to line.")
                    
            # 2. Drive normally using PID if no symbol is acting
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
    stream_buffer = SharedFrameBuffer() 
    
    shared_state = SharedState()
    
    streamer = WebStreamer(stream_buffer)
    streamer.start(port=5000)

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
        
        while cli.app_running:
            raw_frame = picam2.capture_array()
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            
            frame_buffer.write(raw_frame)
            display_frame = raw_frame.copy()

            SHOW_LINE_BOX = True   
            SHOW_SYMBOL_BOX = True  
            
            cv2.line(display_frame, (0, 320), (config.FRAME_WIDTH, 320), (255, 255, 255), 1)

            # 1. DRAW LINE TRACKING
            draw_data = shared_state.get_draw_data()
            if draw_data is not None:
                contour, color, cx, cy = draw_data
                bottom_display_roi = display_frame[240:480, :]
                
                cv2.drawContours(bottom_display_roi, [contour], -1, color, 3)
                
                if SHOW_LINE_BOX:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(bottom_display_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                cv2.circle(bottom_display_roi, (cx, cy), 6, (255, 0, 255), -1)

            # 2. DRAW SYMBOLS
            sym_name, sym_box = shared_state.get_ui_symbol()
            
            if SHOW_SYMBOL_BOX and sym_name and sym_box is not None:
                x, y, w, h = sym_box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.rectangle(display_frame, (x, y - 25), (x + len(sym_name) * 15, y), (255, 255, 0), -1)
                cv2.putText(display_frame, sym_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            frame_height, frame_width, _ = display_frame.shape
            cv2.putText(display_frame, f"Stream: {frame_width}x{frame_height}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            stream_buffer.write(display_frame)
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

if __name__ == "__main__":
    main()