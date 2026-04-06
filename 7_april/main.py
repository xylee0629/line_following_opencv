import numpy as np
from picamera2 import Picamera2
import time, cv2

import config
from motor_control import MotorController
from vision import VisionAnalyser

# Initialisation of picam
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (config.FRAME_WIDTH, config.FRAME_HEIGHT)}))
picam2.start()

time.sleep(1)

# Initialisation of motors


# Load symbol and arrow templates 

# state machine definition

