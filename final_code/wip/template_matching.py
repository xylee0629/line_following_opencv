import cv2
import numpy as np
from picamera2 import Picamera2

frame_width = 640
frame_height = 360

# Do online version first
SYMBOL_PATH = "/home/raspberrypi/line_following_opencv/final_code/images/shapes.png"
symbol_img = cv2.imread(SYMBOL_PATH)

while True:
    gray = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]

    cv2.imshow("Black & White", thresh)
    key = cv2.waitKey(1) & 0xFF	
    if key == ord("q"):
        break
