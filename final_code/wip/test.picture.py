from picamera2 import Picamera2
import time 
import cv2
import numpy as np
import gpiozero

picam2 = Picamera2()
config = picam2.create_still_configuration(main={"format": "RGB888", "size": (640, 360)})
picam2.configure(config)
picam2.start()
time.sleep(2)
"""picam2.preview_configuration.main.size = (640,360)
picam2.configure("preview")"""

while True:
    image = picam2.capture_array()
    roi = image[200:250, 0:639]
    blackline = cv2.inRange(roi, (0,0,0), (100,100,100))
    kernel = np.ones((3,3), np.uint8)
    blackline = cv2.erode(blackline, kernel, iterations=2)
    blackline = cv2.dilate(blackline, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 :
        x,y,w,h = cv2.boundingRect(contours[0])	   
        cv2.line(image, [int(x+(w/2)), 200], [int(x+(w/2)), 250],(255,0,0),3)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow("With contours", image)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
    
picam2.stop()
cv2.destroyAllWindows()
