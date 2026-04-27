import time 
import cv2
import multiprocessing as mp
from multiprocessing import shared_memory
# from picamera2 import Picamera2 (Not used for laptop webcam)

# Import modules
from config import *
from line_worker import line_worker
from img_worker import image_worker
from motor import MotorController

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

    # line_worker shared values
    out_pid = mp.Value('d', 0.0)
    out_reset_pid = mp.Value('b', False)
    out_cx = mp.Value('i', X_CENTRE)
    out_cy = mp.Value('i', Y_CENTRE)
    out_turn_cmd = mp.Value('i', 0)

    # img_worker shared values

    # Initialises event 
    line_ready_event = mp.Event()
    img_ready_event = mp.Event()
    
    # Initialise multiprocessing process workers. 
    # frame shm is passed into the workers
    p_line_worker = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, line_ready_event, out_pid, out_reset_pid, out_cx, out_cy, out_turn_cmd, out_is_priority), 
        daemon=True,
        name="LineWorker"
        )
    p_img_worker = mp.Process(
        target=image_worker , 
        args=(shm.name, frame_lock, img_ready_event, out_is_priority),
        daemon=True,
        name="ImgWorker"
    )

    # Start process
    p_line_worker.start()

    # Initialise camera 
    """
    # Raspberrypi camera
    picam = Picamera2()
    configuration = picam.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
    picam.configure(configuration)
    picam.start()
    time.sleep(2)
    """

    # Laptop webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam does not exist.")
        exit()
    # Set frame size 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    time.sleep(2)
    print("Camera ready.")

    motor = MotorController()

    def resetVision():
        motor.stop()
        out_reset_pid.value = True
        for _ in range(3):
            try:
                cap.read()
            except:
                pass

    try:
        while True:
            ret, frame = cap.read() # Frames are captured in BGR for cap.read()

            # Ensures one process is only accessing frame, then copy to frame buffer
            with frame_lock:
                np.copyto(frame_bf, frame)

            # Frame buffer is copied, event triggers for line_workers
            line_ready_event.set()

            # Process output
            pid = out_pid.value
            cx = out_cx.value
            cy = out_cy.value

            # Junction commands
            turn_cmd = out_turn_cmd.value
            # if turn_cmd is left
            if turn_cmd == 1:
                # Force turn left to enter the junction
                MotorController.move(motor, 55, -55)
                time.sleep(0.8) # Turn for 0.8 seconds
            elif turn_cmd == 2:
                MotorController.move(motor, -55, 55)
                time.sleep(0.8)
            out_turn_cmd.value = 0
            resetVision()

            # Symbols command


    except:
        if cv2.waitKey(1) == ord('q'):
            print("Exiting program.")

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()