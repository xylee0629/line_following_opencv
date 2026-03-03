import cv2 
import numpy as np


# Import img to be read and identified 
img = cv2.imread('/home/raspberrypi/line_following_opencv/images/shapes.png')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



