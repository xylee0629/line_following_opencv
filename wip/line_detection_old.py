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
    # roi = image[200:250, 0:639]
    blackline = cv2.inRange(image, (0,0,0), (100,100,100))
    kernel = np.ones((3,3), np.uint8)
    blackline = cv2.erode(blackline, kernel, iterations=2)
    blackline = cv2.dilate(blackline, kernel, iterations=5)
    contours, hierarchy = cv2.findContours(blackline.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        blackbox = cv2.minAreaRect(contours[0])
        (x_min, y_min), (w_min, h_min), ang = blackbox
        if ang < -45:
            ang = 90 + ang
        if w_min < h_min and ang > 0:	  
            ang = (90-ang)*-1
        if w_min > h_min and ang < 0:
            ang = 90 + ang
        setpoint = 320
        error = int(x_min - setpoint) 
        ang = int(ang)	 
        box = cv2.boxPoints(blackbox)
        box = np.intp(box)
        cv2.drawContours(image,[box],0,(0,0,255),3)	 
        cv2.putText(image,str(ang),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image,str(error),(10, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.line(image, (int(x_min),200 ), (int(x_min),250 ), (255,0,0),3)
        
        cv2.imshow("orginal with line", image)
        
        key = cv2.waitKey(1) & 0xFF	
        if key == ord("q"):
            break
    
picam2.stop()
cv2.destroyAllWindows()
