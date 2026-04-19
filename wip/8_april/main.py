import time
import threading
from picamera2 import Picamera2
import config
from motor_control import MotorController
from streamer import WebStreamer
from cli_manager import TerminalController
from state import SharedState
from vision import VisionAnalyzer

def thread_camera(state, cli):
    try:
        picam2 = Picamera2()
        cam_config = picam2.create_video_configuration(
            main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"} 
        )
        picam2.configure(cam_config)
        picam2.start()
        print("[SYSTEM] Camera initialized successfully.")
        
        while cli.app_running:
            frame = picam2.capture_array()
            with state.lock:
                state.latest_frame = frame
            time.sleep(0.01) 
            
        picam2.stop()
    except Exception as e:
        print(f"\n[ERROR] Camera Thread Crashed: {e}")

def thread_line_follow(vision, motors, frame_buffer, shared_state, cli):
    # Initialize the memory flags for when the camera loses the line
    left_flag, right_flag = 0, 0
    
    while cli.app_running:
        frame = frame_buffer.read()
        if frame is not None:
            bottom_roi = frame[320:480, :]
            
            # --- NEW LOGIC: Call process_line instead of get_line_paths ---
            # This handles the Red -> Yellow -> Black priority AND the centroid math all at once
            cx, left_flag, right_flag, draw_data = vision.process_line(bottom_roi, left_flag, right_flag)
            
            # Calculate PID and update steering
            left_pwm, right_pwm = motors.calculate_pid(cx)
            shared_state.update_steering(left_pwm, right_pwm, draw_data)
            
        time.sleep(0.01)

def thread_motor_movement(state, cli, motors):
    """PID control with dynamic braking for curves and junctions."""
    while cli.app_running:
        if cli.robot_active:
            with state.lock:
                current_cx = state.cx
                current_path_count = state.path_count
            
            error = abs(current_cx - config.FRAME_CENTRE)
            
            # State Machine for Motor Profiling
            if current_path_count >= 2:
                # JUNCTION: Slow hard turn!
                motors.set_profile(config.FREQUENCY, 0.20) 
            elif error > 50: 
                # SHARP CURVE: Brake slightly
                motors.set_profile(config.FREQUENCY, 0.25)
            else:
                # STRAIGHT: Cruise
                motors.set_profile(config.FREQUENCY, config.DUTY_CYCLE)
                
            left_pwm, right_pwm = motors.calculate_pid(current_cx)
            motors.move(left_pwm, right_pwm)
            
        else:
            motors.stop()
            
        time.sleep(0.01)

def thread_symbol_detection(state, cli, vision):
    """(FUTURE) Scans the top of the frame for symbols/arrows."""
    while cli.app_running:
        # Example of how to structure this when you are ready:
        # with state.lock:
        #     frame = state.latest_frame
        # if frame is not None:
        #     top_roi = frame[0: int(config.FRAME_HEIGHT*0.3), :]
        #     symbol, box = vision.detect_symbol(top_roi)
        #     if symbol in ["ARROW_LEFT", "ARROW_RIGHT"]:
        #         with state.lock:
        #             state.pending_turn = symbol
        time.sleep(0.1) 

if __name__ == "__main__":
    state = SharedState()
    cli = TerminalController()
    motors = MotorController()
    vision = VisionAnalyzer() 
    streamer = WebStreamer(state)
    
    cli.start()
    streamer.start(port=5000)
    
    cam_thread = threading.Thread(target=thread_camera, args=(state, cli), daemon=True)
    vision_thread = threading.Thread(target=thread_line_follow, args=(state, cli, vision), daemon=True)
    motor_thread = threading.Thread(target=thread_motor_movement, args=(state, cli, motors), daemon=True)
    symbol_thread = threading.Thread(target=thread_symbol_detection, args=(state, cli, vision), daemon=True)
    
    cam_thread.start()
    vision_thread.start()
    motor_thread.start()
    symbol_thread.start()
    
    try:
        while cli.app_running:
            time.sleep(1)
    except KeyboardInterrupt:
        cli.app_running = False
        print("\n[SYSTEM] Shutting down...")
    finally:
        motors.cleanup()