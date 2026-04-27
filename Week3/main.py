import time
import cv2 as cv
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from picamera2 import Picamera2

# Import from modules
from Week3.config import *
from Week3.motor import MotorController 
from vision_utils import _read_str, _write_str
from Week3.line_worker import line_worker
from Week3.img_worker import image_worker
from webstreamer import run_streamer 

# ══════════════════════════════════════════════════════════════
# MOTOR HELPER
# ══════════════════════════════════════════════════════════════
def set_robot_speed(controller, left_speed, right_speed):
    """Converts 0-100 speed scale to 0.0-1.0 and clamps bounds for gpiozero."""
    l_val = max(-1.0, min(1.0, left_speed / 100.0))
    r_val = max(-1.0, min(1.0, right_speed / 100.0))
    controller.move(l_val, r_val)

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    shm  = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    line_disp_shm = shared_memory.SharedMemory(create=True, size=LINE_DISPLAY_BYTES)
    img_disp_shm  = shared_memory.SharedMemory(create=True, size=IMG_DISPLAY_BYTES)
    line_disp_buf = np.ndarray(LINE_DISPLAY_SHAPE, dtype=np.uint8, buffer=line_disp_shm.buf)
    img_disp_buf  = np.ndarray(IMG_DISPLAY_SHAPE,  dtype=np.uint8, buffer=img_disp_shm.buf)
    line_disp_lock, img_disp_lock = mp.Lock(), mp.Lock()

    frame_lock = mp.Lock()
    
    # 🌟 NEW: Locks and Events for concurrency and memory safety
    string_lock = mp.Lock()
    line_ready_event = mp.Event()
    img_ready_event = mp.Event()
    out_reset_pid = mp.Value('b', False)

    out_pid = mp.Value('d', 0.0); out_cx = mp.Value('i', X_CENTRE); out_cy = mp.Value('i', Y_CENTRE)
    out_lineArea = mp.Value('d', 0.0); out_has_line = mp.Value('b', False)
    
    out_found = mp.Value('b', False); out_label = mp.Array('c', 64)
    out_instruction = mp.Array('c', 32); out_instruction_ready = mp.Value('b', False) 
    out_is_priority = mp.Value('b', False); out_turn_cmd = mp.Value('i', 0) 
    out_investigating = mp.Value('b', False)

    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, line_ready_event, out_reset_pid, out_pid, out_cx, out_cy, out_has_line, out_lineArea, out_is_priority, out_turn_cmd, line_disp_shm.name, line_disp_lock),
        daemon=True, name="LineWorker"
    )
    p_img = mp.Process(
        target=image_worker,
        args=(shm.name, frame_lock, img_ready_event, string_lock, out_found, out_label, out_instruction, out_instruction_ready, img_disp_shm.name, img_disp_lock, out_is_priority, out_investigating),
        daemon=True, name="ImgWorker"
    )
    
    p_stream = mp.Process(
        target=run_streamer,
        args=(line_disp_shm.name, LINE_DISPLAY_SHAPE, line_disp_lock, img_disp_shm.name, IMG_DISPLAY_SHAPE, img_disp_lock),
        daemon=True, name="WebStreamer"
    )

    p_line.start(); p_img.start(); p_stream.start()
    print("[main] Workers and Streamer started.")

    robot_motors = MotorController()

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
    picam2.start()
    time.sleep(2.0)                       
    print("[main] Camera ready. Web server is running. Press Ctrl+C to quit.")

    # 🌟 NEW: Helper function to clear hardware buffer after a sleep() maneuver
    def reset_vision():
        robot_motors.stop()
        out_reset_pid.value = True
        for _ in range(3): 
            try: picam2.capture_array()
            except: pass

    line_loss_counter = 0
    active_instruction = ""
    
    try:
        while True:
            RGB = picam2.capture_array()
            if RGB.ndim == 3 and RGB.shape[2] == 4: RGB = RGB[:, :, :3]

            with frame_lock: np.copyto(fbuf, RGB)
            
            # Wake up the multiprocessing workers instantly
            line_ready_event.set()
            img_ready_event.set()

            pid, cx, cy = out_pid.value, out_cx.value, out_cy.value
            has_line, found = bool(out_has_line.value), bool(out_found.value)
            
            # Safely read strings using the lock
            label = _read_str(out_label, string_lock)
            current_is_priority = bool(out_is_priority.value)
            
            new_instr= ""
            if out_instruction_ready.value:
                new_instr = _read_str(out_instruction, string_lock)
                out_instruction_ready.value = False   

            # ══════════════════════════════════════════════════════════
            # ACTION PRIORITY 1: JUNCTION EXITS
            # ══════════════════════════════════════════════════════════
            turn_cmd = out_turn_cmd.value
            if turn_cmd != 0:
                if turn_cmd == 1:
                    print("[main] Exiting priority line — turning LEFT")
                    # 🌟 FIX: Balanced motors to prevent stalling
                    set_robot_speed(robot_motors, -55, 55); time.sleep(0.8)
                elif turn_cmd == 2:
                    print("[main] Exiting priority line — turning RIGHT")
                    # 🌟 FIX: Balanced motors to prevent stalling
                    set_robot_speed(robot_motors, 55, -55); time.sleep(0.8)
                out_turn_cmd.value = 0
                reset_vision()

            if new_instr:
                active_instruction = new_instr
                print(f"[main] Instruction '{active_instruction}' confirmed.")
                _write_str(out_instruction, "", 32, string_lock)   

            # ══════════════════════════════════════════════════════════
            # ACTION PRIORITY 2: SHAPE INSTRUCTIONS
            # ══════════════════════════════════════════════════════════
            if active_instruction:
                if active_instruction == "TURN_LEFT":
                    set_robot_speed(robot_motors, MOTOR_LEFT_NORMAL - pid, MOTOR_RIGHT_NORMAL + pid); time.sleep(0.6)
                    # 🌟 FIX: Increased turn speed to overcome friction
                    set_robot_speed(robot_motors, -55, 55); time.sleep(0.1)
                    active_instruction = ""
                    reset_vision()
                elif active_instruction == "TURN_RIGHT":
                    set_robot_speed(robot_motors, MOTOR_LEFT_NORMAL - pid, MOTOR_RIGHT_NORMAL + pid); time.sleep(0.7)
                    # 🌟 FIX: Increased turn speed to overcome friction
                    set_robot_speed(robot_motors, 55, -55); time.sleep(0.1)
                    active_instruction = ""
                    reset_vision()
                elif active_instruction == "MOVE_FORWARD":
                    set_robot_speed(robot_motors, MOTOR_LEFT_NORMAL, MOTOR_RIGHT_NORMAL); time.sleep(0.2)
                    active_instruction = ""
                    reset_vision()
                elif active_instruction == "STOP":
                    robot_motors.stop(); time.sleep(2)
                    set_robot_speed(robot_motors, 50, 50); time.sleep(0.2)
                    active_instruction = ""
                    reset_vision()
                elif active_instruction == "360-TURN":
                    set_robot_speed(robot_motors, MOTOR_LEFT_NORMAL, MOTOR_RIGHT_NORMAL); time.sleep(0.5)
                    set_robot_speed(robot_motors, -70, 70); time.sleep(2.1)
                    active_instruction = ""
                    reset_vision()
            
            # ══════════════════════════════════════════════════════════
            # ACTION PRIORITY 3: STANDARD LINE FOLLOWING
            # ══════════════════════════════════════════════════════════
            else:
                if current_is_priority:
                    # 🌟 FIX: Bumped up slightly so it doesn't stall on the Red/Yellow line
                    L_base, R_base = 35, 35
                else:
                    L_base = MOTOR_LEFT_PUSH if found else MOTOR_LEFT_NORMAL
                    R_base = MOTOR_RIGHT_PUSH if found else MOTOR_RIGHT_NORMAL

                if has_line:
                    left_cmd = L_base - pid
                    right_cmd = R_base + pid
                    
                    # 🌟 FIX: Anti-Stall Deadband Filter
                    # Stops the inner wheel from getting stuck at e.g., 10% power 
                    # and dragging during sharp PID adjustments.
                    DEADBAND = 25
                    
                    if 0 < left_cmd < DEADBAND: left_cmd = 0
                    elif -DEADBAND < left_cmd <= 0: left_cmd = -DEADBAND
                    
                    if 0 < right_cmd < DEADBAND: right_cmd = 0
                    elif -DEADBAND < right_cmd <= 0: right_cmd = -DEADBAND

                    set_robot_speed(robot_motors, left_cmd, right_cmd)
                    line_loss_counter = 0
                else:
                    # 🌟 FIX: Reduced line_loss_counter from 8 to 2.
                    # Prevents massive sweeping arcs by initiating spins faster.
                    if line_loss_counter <= 2:
                        set_robot_speed(robot_motors, MOTOR_LEFT_NORMAL, MOTOR_RIGHT_NORMAL)
                        line_loss_counter += 1
                    else:
                        # 🌟 FIX: Balanced recovery spin speeds
                        if pid > 0: set_robot_speed(robot_motors, -55, 55)
                        else: set_robot_speed(robot_motors,  55, -55)
                    
            # ══════════════════════════════════════════════════════════
            # UI UPDATES
            # ══════════════════════════════════════════════════════════
            if active_instruction:
                with line_disp_lock:
                    cv.putText(line_disp_buf, f"CMD: {active_instruction}", (10, 160), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                with img_disp_lock:
                    cv.putText(img_disp_buf, f"CMD: {active_instruction}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    except KeyboardInterrupt: print("\n[main] Ctrl+C received — stopping.")
    except Exception as e:
        import traceback; print(f"\n[main] ERROR: {e}"); traceback.print_exc()
    finally:
        print("[main] Shutting down…")
        
        try: robot_motors.cleanup()
        except: pass

        # Cleanly terminate multiprocessing workers
        p_line.terminate(); p_line.join()
        p_img.terminate(); p_img.join()
        p_stream.terminate(); p_stream.join() 

        # Wipe shared memory buffers
        try: 
            shm.close(); shm.unlink() 
            line_disp_shm.close(); line_disp_shm.unlink() 
            img_disp_shm.close(); img_disp_shm.unlink()
        except: pass

        try: picam2.stop()
        except: pass

        cv.destroyAllWindows()
        print("[main] Done.")

if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()