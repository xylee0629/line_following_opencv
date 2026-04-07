# main.py
import cv2
import time
import threading
from picamera2 import Picamera2

import config
from motor_control import MotorController
from streamer import WebStreamer
from cli_manager import TerminalController
from state import SharedState
from vision import Vision

def thread_camera(state, cli):
    """Continuously captures frames from the Pi Camera and updates the shared state."""
    picam2 = Picamera2()
    cam_config = picam2.create_video_configuration(
        main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"} 
    )
    picam2.configure(cam_config)
    picam2.start()
    
    while cli.app_running:
        """Capture frame, convert to BGR for OpenCV consistency, and store it"""
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        state.latest_frame = frame
        time.sleep(0.01) # Small delay to yield resources
        
    picam2.stop()

def thread_line_follow(state, cli, vision, motors):
    """Processes the latest frame for line tracking and calculates PID steering."""
    while cli.app_running:
        frame = state.read()
        if frame is not None:
            # Look for a symbol detected by the symbol thread (defaults to None)
            symbol = getattr(state, "symbol", None)
            
            # Find the path's target center (cx)
            cx = vision.line_processing(frame, symbol)
            
            # Calculate motor PWMs using the MotorController's PID logic
            left_pwm, right_pwm = motors.calculate_pid(cx)
            
            # Update the shared steering state 
            state.steering = (left_pwm, right_pwm)
            
        time.sleep(0.02) # Cap at ~50Hz

def thread_symbol_detection(state, cli):
    """Scans the current frame for signs/symbols and updates the state."""
    while cli.app_running:
        frame = state.read()
        if frame is not None:
            # Placeholder: Implement your QR code or template matching logic here
            # using config.SYMBOL_PATHS. Once a symbol is found, update state.
            state.symbol = None 
            
        time.sleep(0.1) # Symbol detection is heavy, run at ~10Hz

def thread_motor_movement(state, cli, motors):
    """Reads the steering instructions and drives the motors if the robot is active."""
    while cli.app_running:
        if cli.robot_active:
            # Read calculated PID outputs
            left_pwm, right_pwm = state.steering
            motors.move(left_pwm, right_pwm)
        else:
            # Paused via terminal controller ('s')
            motors.stop()
            
        time.sleep(0.01)
        
    # Ensure motors are completely off when the app quits
    motors.cleanup()

if __name__ == "__main__":
    # 1. Initialize all components
    state = SharedState()
    cli = TerminalController()
    motors = MotorController()
    vision = Vision()
    
    # The streamer needs an object with a .read() method that returns frames (SharedState fits perfectly)
    streamer = WebStreamer(state)
    
    # 2. Start standalone daemon services
    cli.start()
    streamer.start(port=5000)
    
    # 3. Define standard threads
    t_cam = threading.Thread(target=thread_camera, args=(state, cli), daemon=True)
    t_line = threading.Thread(target=thread_line_follow, args=(state, cli, vision, motors), daemon=True)
    t_sym = threading.Thread(target=thread_symbol_detection, args=(state, cli), daemon=True)
    t_motor = threading.Thread(target=thread_motor_movement, args=(state, cli, motors), daemon=True)
    
    # 4. Start standard threads
    print("[SYSTEM] Starting threads...")
    t_cam.start()
    t_line.start()
    t_sym.start()
    t_motor.start()
    
    # 5. Keep the main process alive until the CLI signals a quit
    try:
        while cli.app_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SYSTEM] KeyboardInterrupt detected. Shutting down...")
        cli.app_running = False
        
    print("[SYSTEM] Joining threads...")
    t_cam.join(timeout=2.0)
    t_line.join(timeout=2.0)
    t_sym.join(timeout=2.0)
    t_motor.join(timeout=2.0)
    
    print("[SYSTEM] Shutdown complete.")
        
