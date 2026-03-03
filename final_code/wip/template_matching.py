# import the necessary packages
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap.add_argument("-i", "--image", required=True,
	help="/home/raspberrypi/line_following_opencv/final_code/images/shapes.png")
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())
# load the image, convert it to grayscale, blur it slightly,
# and threshold it
image = cv2.imread(args["image"])

while True:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)[1]
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        cv2.imshow("Image", image)

    key = cv2.waitKey(1) & 0xFF	
    if key == ord("q"):
        break
