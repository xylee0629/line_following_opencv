import cv2
import numpy as np
from picamera2 import Picamera2

ref_img = cv2.imread('/home/raspberrypi/line_following_opencv/images/shapes.png')
ref_copy = ref_img.copy()

# --- NEW: Use HSV Saturation instead of Grayscale ---
ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
_, ref_saturation, _ = cv2.split(ref_hsv) # Extract only the S channel

# Threshold the Saturation channel (Notice we do NOT use _INV here, 
# because in the S channel, shapes are already bright and background is dark)
ret, ref_threshold = cv2.threshold(ref_saturation, 50, 255, cv2.THRESH_BINARY)

ref_contours, hierarchy = cv2.findContours(ref_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

templates = []
shape_name = [
    "Star",
    "Diamond",
    "Cross",
    "Trapezoid",
    "3/4 Semicircle", 
    "Semicircle", 
    "Octagon"
]

valid_contours = [c for c in ref_contours if 100 < cv2.contourArea(c) < (ref_img.shape[0]*ref_img.shape[1]*0.9)]
valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0]))

print("\n--- REFERENCE SHAPES HU MOMENTS ---")
for i, cnt in enumerate(valid_contours):
    if i < len(shape_name):
        name = shape_name[i]
    else:
        name = f"Extra Shape {i}"
        
    # --- NEW: Calculate Solidity for the template ---
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0
    # ------------------------------------------------
    
    templates.append({"name": name, "contour": cnt, "solidity": solidity})
    
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(ref_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(ref_copy, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
print(f"\nLearned {len(templates)} shapes from the reference image.")    


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 360)}))
picam2.start()

print("\nStarting live feed. Press 'q' to quit.")
print("--- LIVE DETECTION LOG ---")

while True:
    frame = picam2.capture_array()
    
    # Preprocessing
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, frame_saturation, _ = cv2.split(frame_hsv)
    frame_blur = cv2.GaussianBlur(frame_saturation, (7, 7), 0)
    
    # Otsu's thresholding
    ret, frame_threshold = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- FIX: Back to RETR_EXTERNAL, and we don't need hierarchy anymore! ---
    frame_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in frame_contours:
        area = cv2.contourArea(cnt)
        if area < 500: # Ignore small noise
            continue
        
        # Calculate Live Shape Solidity
        live_hull = cv2.convexHull(cnt)
        live_hull_area = cv2.contourArea(live_hull)
        live_solidity = area / live_hull_area if live_hull_area > 0 else 0

        best_match_name = "Unknown"
        best_match_score = 0.50 # Combined threshold score
        
        for template in templates:
            # 1. Get Hu Moments Score
            hu_score = cv2.matchShapes(template["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
            
            # 2. Get Solidity Difference
            solidity_diff = abs(template["solidity"] - live_solidity)
            
            # 3. Combine them
            total_score = hu_score + (solidity_diff * 2.0)
            
            if total_score < best_match_score:
                best_match_score = total_score
                best_match_name = template["name"]
        
        # Draw bounding box and label
        if best_match_name != "Unknown":
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            label = f"{best_match_name} ({best_match_score:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the feeds
    cv2.imshow("Reference", ref_copy)
    cv2.imshow("Threshold (Saturation)", frame_threshold) 
    cv2.imshow("Live Shape Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
picam2.stop()       
cv2.destroyAllWindows()