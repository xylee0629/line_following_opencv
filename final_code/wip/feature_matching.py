import cv2
import numpy as np
from picamera2 import Picamera2

# 1. Load the reference symbol image
# Replace 'symbol.png' with the path to your reference image
SYMBOL_PATH = '/home/raspberrypi/line_following_opencv/final_code/images/star.png'
symbol_img = cv2.imread(SYMBOL_PATH, cv2.IMREAD_GRAYSCALE)

if symbol_img is None:
    print(f"Error: Could not load the image at {SYMBOL_PATH}")
    exit()

# 2. Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=500) # Limit features to save CPU on the Pi

# Find the keypoints and descriptors for the reference symbol
kp_symbol, des_symbol = orb.detectAndCompute(symbol_img, None)

# 3. Initialize the Feature Matcher
# Brute Force Matcher with Hamming distance (best for ORB)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Minimum number of matches needed to confidently say we found the symbol
MIN_MATCH_COUNT = 15 

# 4. Initialize PiCamera2
picam2 = Picamera2()
# Configure the camera for a lower resolution to keep frame rates high
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("Camera started. Press 'q' to quit.")

try:
    while True:
        # Capture a frame directly as a numpy array
        frame = picam2.capture_array()
        
        # Convert the frame to grayscale for feature matching
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find keypoints and descriptors in the live frame
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
        
        # Ensure we found descriptors in the frame before matching
        if des_frame is not None and len(des_frame) > 0:
            # Match descriptors between the symbol and the live frame
            matches = bf.match(des_symbol, des_frame)
            
            # Sort them in the order of their distance (lower distance = better match)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # If we have enough good matches, find the object!
            if len(matches) >= MIN_MATCH_COUNT:
                # Extract the coordinates of the matching points
                src_pts = np.float32([kp_symbol[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Find the Homography matrix (perspective transformation)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    # Get the dimensions of the reference symbol
                    h, w = symbol_img.shape
                    
                    # Define the corners of the reference symbol
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    
                    # Project the corners into the live frame using the Homography matrix
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Draw a green bounding box around the detected symbol
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                # Optional: print out how many matches were found if it's below the threshold
                print(f"Not enough matches: {len(matches)}/{MIN_MATCH_COUNT}")
                pass

        # Optional: You can draw the raw matches for debugging purposes
        match_img = cv2.drawMatches(symbol_img, kp_symbol, frame, kp_frame, matches[:15], None, flags=2)
        cv2.imshow('Feature Matches', match_img)

        # Display the live frame with the bounding box
        cv2.imshow('Symbol Tracker', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    # Always clean up resources
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped and windows closed.")
