# main.py
import cv2
import time
from picamera2 import Picamera2

import config
from motor_control import MotorController
from vision import VisionAnalyzer
from cli_manager import TerminalController


class RobotState:
    IDLE = "IDLE"
    LINE_FOLLOW = "LINE_FOLLOW"
    DETECT_SYMBOL = "DETECT_SYMBOL"
    EXECUTE_ACTION = "EXECUTE_ACTION"
    STOP = "STOP"

def main():
    # Initialize Modules
    motors = MotorController()
    vision = VisionAnalyzer()
    cli = TerminalController()
    
    # Initialize Camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"format": 'BGR888', "size": (config.FRAME_WIDTH, config.FRAME_HEIGHT)}))
    picam2.start()
    time.sleep(1) # Allow camera sensor to warm up

    left_flag, right_flag = 0, 0
    frame_count = 0
    
    # FSM Variables
    current_state = RobotState.IDLE
    detected_symbol = None 

    # Start CLI Listener (Background thread for terminal commands)
    cli.start()

    try:
        while cli.app_running:
            # Capture frame at the top of the loop for all states
            frame = picam2.capture_array()
            top_roi = frame[0:400, :]
            bottom_roi = frame[240:480, :]

            # ==========================================
            # STATE MACHINE EXECUTION
            # ==========================================
            if current_state == RobotState.IDLE:
                motors.stop()
                if cli.robot_active:
                    print("\n[STATE] Switching to LINE_FOLLOW")
                    current_state = RobotState.LINE_FOLLOW

            elif current_state == RobotState.LINE_FOLLOW:
                # 1. Check for manual pause from terminal
                if not cli.robot_active:
                    current_state = RobotState.IDLE
                    continue

                # 2. Vision Math (Abstracted line tracking logic)
                cx, left_flag, right_flag = vision.process_line(bottom_roi, left_flag, right_flag)
                        
                # 3. Compute Steering (PID) and Move
                left_pwm, right_pwm = motors.calculate_pid(cx)
                motors.move(left_pwm, right_pwm)
                
                # 4. State Transition Check (Every 3rd frame, check for symbols)
                frame_count += 1
                if frame_count % 3 == 0:
                    current_state = RobotState.DETECT_SYMBOL

            elif current_state == RobotState.DETECT_SYMBOL:
                # 1. Vision Math (Abstracted ORB/Arrow logic with new padding/safety guards)
                detected_symbol = vision.detect_symbol(top_roi)

                # 2. State Transition Logic based on findings
                if detected_symbol:
                    print(f"\n[STATE] Symbol Detected: {detected_symbol} -> Switching to EXECUTE_ACTION")
                    current_state = RobotState.EXECUTE_ACTION
                else:
                    current_state = RobotState.LINE_FOLLOW

            elif current_state == RobotState.EXECUTE_ACTION:
                # Perform the required action based on the symbol
                if detected_symbol == "Hazard":
                    print("[ACTION] Hazard detected! Proceeding to STOP state.")
                    current_state = RobotState.STOP
                    
                elif detected_symbol == "Button":
                    print("[ACTION] Button detected. Pausing for 2 seconds.")
                    motors.stop()
                    time.sleep(2.0)
                    current_state = RobotState.LINE_FOLLOW
                    
                elif str(detected_symbol).startswith("ARROW_"):
                    print(f"[ACTION] Navigating based on {detected_symbol}.")
                    # Add turning logic here in the future
                    current_state = RobotState.LINE_FOLLOW
                    
                else:
                    print(f"[ACTION] Recognized {detected_symbol}, logging and continuing.")
                    current_state = RobotState.LINE_FOLLOW

                # Clear memory for the next loop
                detected_symbol = None

            elif current_state == RobotState.STOP:
                motors.stop()
                cli.robot_active = False # Deactivate terminal flag
                print("\n[STATE] Robot STOPPED. Type 's' + Enter to restart.")
                # Transition back to IDLE to wait for user input
                current_state = RobotState.IDLE 

            # Display feed (Optional for debugging)
            cv2.imshow("Robot View", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\nProgram stopped by Ctrl+C") 
    finally:
        cli.app_running = False
        motors.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

