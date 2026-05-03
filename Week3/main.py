import time 
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
from picamera2 import Picamera2 

# Import modules
from config import *
from line_worker import line_worker
from img_worker import image_worker
from webstreamer import run_streamer
from vision_utils import _read_str, _write_str
from motor import move, stop, cleanup

# Functions executed by process are defined in module files and imported into main.py for readability.

def main():
    """
    Initialise shared memory for frame. 
    Creates a shared memory block for all cores to access.
    """
    shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
    frame_bf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf) # Creates a frame buffer to be copied into
    out_is_priority = mp.Value('b', False) # Checks if currently on red/yellow line

    # Locks shared memory so that one process accesses values one at a time to avoid race conditions
    frame_lock = mp.Lock()
    string_lock = mp.Lock() # Ensures writing complete string

    # line_worker shared values
    line_display_shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
    line_display_bf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=line_display_shm.buf)
    line_display_lock = mp.Lock()

    out_pid = mp.Value('d', 0.0)
    out_reset_pid = mp.Value('b', False)
    out_turn_cmd = mp.Value('i', 0)
    out_has_line = mp.Value('b', False)

    # img_worker shared values
    img_display_shm = shared_memory.SharedMemory(create=True, size=FRAME_BYTES)
    img_display_bf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=img_display_shm.buf)
    img_display_lock = mp.Lock()

    out_found = mp.Value('b', False)
    out_instruction = mp.Array('c', 32)
    out_instruction_ready = mp.Value('b', False) 

    # Initialises event 
    line_ready_event = mp.Event()
    img_ready_event = mp.Event()
    
    # Initialise multiprocessing process workers. 
    # frame shm is passed into the workers
    p_line_worker = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, line_ready_event, out_pid, out_reset_pid, out_turn_cmd, out_is_priority, out_has_line, line_display_shm.name, line_display_lock), 
        daemon=True,
        name="LineWorker"
        )
    p_img_worker = mp.Process(
        target=image_worker , 
        args=(shm.name, frame_lock, img_ready_event, string_lock, out_is_priority, out_found, out_instruction, out_instruction_ready, img_display_shm.name, img_display_lock),
        daemon=True,
        name="ImgWorker"
    )
    p_stream_worker = mp.Process(
        target=run_streamer ,
        args=(line_display_shm.name, FRAME_SHAPE, line_display_lock, img_display_shm.name, FRAME_SHAPE, img_display_lock),
        daemon=True,
        name="WebStreamer"
    )

    # Start process
    p_line_worker.start()
    p_img_worker.start()
    p_stream_worker.start()

    def resetVision():
        stop()
        out_reset_pid.value = True
        for _ in range(3):
            try:
                picam.capture_array()
            except:
                pass

    line_loss_counter = 0
    active_instruction = ""

    try:
        # Initialise camera 
        # Raspberrypi camera
        picam = Picamera2()
        configuration = picam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
        picam.configure(configuration)
        picam.start()
        time.sleep(2)
        print("Camera ready.")

        while True:
            RGB = picam.capture_array()
            if RGB.ndim == 3 and RGB.shape[2] == 4: RGB = RGB[:, :, :3]

            # Ensures one process is only accessing frame, then copy to frame buffer
            with frame_lock:
                np.copyto(frame_bf, RGB)

            # Frame buffer is copied, event triggers for line_workers
            line_ready_event.set()
            img_ready_event.set()

            # Process output
            pid = out_pid.value
            """cx = out_cx.value
            cy = out_cy.value"""
            current_priority = bool(out_is_priority.value)
            has_line = bool(out_has_line.value)
            found = bool(out_found.value)
            

            new_instruction = ""
            if out_instruction_ready.value:
                new_instruction = _read_str(out_instruction, string_lock)
                out_instruction_ready.value = False   

            # Priority 1: Junction command 
            turn_cmd = out_turn_cmd.value
            # if turn_cmd is left
            if turn_cmd == 1:
                # Force turn left to enter the junction
                move(55, -55)
                time.sleep(0.8) # Turn for 0.8 seconds
            elif turn_cmd == 2:
                move(-55, 55)
                time.sleep(0.8)
            out_turn_cmd.value = 0
            resetVision()

            # Symbols command
            # Check for new instruction for active 
            if new_instruction:
                active_instruction = new_instruction
                _write_str(out_instruction, "", 32, string_lock)

            # Priority 2: Execute instruction
            if active_instruction:
                if active_instruction == "TURN_LEFT":
                    move(MOTOR_BASE_SPEED - pid, MOTOR_BASE_SPEED + pid)
                    time.sleep(0.6)
                    move(-55, 55)
                    time.sleep(0.1)
                    active_instruction = ""
                    resetVision()

                elif active_instruction == "TURN_RIGHT":
                    move(MOTOR_BASE_SPEED - pid, MOTOR_BASE_SPEED + pid)
                    time.sleep(0.7)
                    move(55, -55)
                    time.sleep(0.1)
                    active_instruction = ""
                    resetVision()

                elif active_instruction == "MOVE_FORWARD":
                    move(MOTOR_BASE_SPEED, MOTOR_BASE_SPEED)
                    time.sleep(0.2)
                    active_instruction = ""
                    resetVision()

                elif active_instruction == "STOP":
                    stop()
                    time.sleep(2)
                    move(50, 50)
                    time.sleep(0.2)
                    active_instruction = ""
                    resetVision()
                elif active_instruction == "360-TURN":
                    move(MOTOR_BASE_SPEED, MOTOR_BASE_SPEED)
                    time.sleep(0.5)
                    move(-70, 70)
                    time.sleep(2.1)
                    active_instruction = ""
                    resetVision()

            # Priority 3: Line-Following
            else:
                if current_priority:
                    L_Base = 35
                    R_Base = 35
                else:
                    if found is True:
                        L_Base = MOTOR_PUSH_SPEED
                        R_Base = MOTOR_PUSH_SPEED
                    else:
                        L_Base = MOTOR_BASE_SPEED
                        R_Base = MOTOR_BASE_SPEED

                if has_line:
                    left_cmd = L_Base - pid
                    right_cmd = R_Base + pid

                    # Prevents duty cycle from going too low 
                    DEADBAND = 25
                    if 0 < left_cmd < DEADBAND: # left cmd with pid is too low
                        left_cmd = 0 # set to 0
                    elif -DEADBAND < left_cmd < 0:
                        left_cmd = -DEADBAND

                    if 0 < right_cmd < DEADBAND:
                        right_cmd = 0
                    elif -DEADBAND < right_cmd < 0:
                        right_cmd = -DEADBAND

                    move(left_cmd, right_cmd)
                    line_loss_counter = 0

                else:
                    if line_loss_counter <= 2:
                        move(MOTOR_BASE_SPEED, MOTOR_BASE_SPEED)
                        line_loss_counter += 1
                    else:
                        if pid > 0:
                            move(-55, 55)
                        else:
                            move(55, -55)

            if active_instruction:
                with line_display_lock:
                    cv2.putText(line_display_bf, f"CMD: {active_instruction}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                with img_display_lock:
                    cv2.putText(img_display_bf, f"CMD: {active_instruction}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    except KeyboardInterrupt:
        print("\n[main] Ctrl+C received — stopping.")

    except Exception as e:
        import traceback
        print(f"\n[main] ERROR: {e}")
        traceback.print_exc()

    finally:
        print("[main] shutting down.")
        try:
            cleanup()
        except:
            pass

        # Terminate all workers
        p_line_worker.terminate()
        p_line_worker.join()

        p_img_worker.terminate()
        p_img_worker.join()

        p_stream_worker.terminate()
        p_stream_worker.join()

        try:
            shm.close()
            shm.unlink()

            line_display_shm.close()
            line_display_shm.unlink()

            img_display_shm.close()
            img_display_shm.unlink()

        except:
            pass

        try:
            picam.close()
        except:
            pass
        
        cv2.destroyAllWindows()
        print("Program successfully stopped.")


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    main()