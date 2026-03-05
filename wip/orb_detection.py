import cv2
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. INITIALIZE ORB & MATCHER
# ==========================================
# Initialize ORB detector
orb = cv2.ORB_create()

# We remove crossCheck=True because Lowe's Ratio Test requires k-Nearest Neighbors (KNN)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) 

MIN_MATCH_COUNT = 8 # Minimum inliers needed to confirm a shape

# ==========================================
# 2. SETUP SYMBOL TEMPLATES (ORB)
# ==========================================
symbol_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/button.jpg', "name": "Button"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/fingerprint.jpg', "name": "Fingerprint"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/hazard.jpg', "name": "Hazard"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/qr.jpg', "name": "QR Code"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/recycle.jpg', "name": "Recycle"}
]

symbol_templates = []
print("\n--- LEARNING REFERENCE SYMBOLS (ORB) ---")

for sym_data in symbol_files:
    file_path = sym_data["filepath"]
    name = sym_data["name"]
    
    # Read as grayscale since ORB only needs intensity/texture, not color
    sym_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if sym_img is None:
        print(f"ERROR: Could not load symbol image at {file_path}. Skipping.")
        continue
        
    print(f"Processing: {file_path}")
    
    # Detect keypoints and compute descriptors for the master template
    kp, des = orb.detectAndCompute(sym_img, None)
    
    if des is not None:
        symbol_templates.append({
            "name": name,
            "kp": kp,
            "des": des
        })
        print(f"  -> Learned: {name} ({len(kp)} keypoints)")
    else:
        print(f"  -> Failed to find keypoints in {name}")

print(f"\nSuccessfully learned {len(symbol_templates)} master symbol templates.")

# ==========================================
# 3. PICAMERA LIVE FEED
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

print("\nStarting live ORB feed. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    
    # Grayscale is required for ORB feature extraction
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # We still use HSV/Saturation to find the physical objects to look at
    # (Running ORB on the whole 640x360 frame every tick is too slow for a Pi)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, frame_saturation, _ = cv2.split(frame_hsv)
    
    frame_blur = cv2.GaussianBlur(frame_saturation, (15, 15), 0) 
    ret, frame_threshold = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((30, 30), np.uint8)
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
    
    if ret < 40:
        frame_threshold = np.zeros_like(frame_threshold)
        
    frame_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in frame_contours:
        area = cv2.contourArea(cnt)
        if area < 500: # Ignore tiny background noise
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        best_match_name = "Unknown Symbol"
        
        # ==========================================
        # ORB FEATURE EXTRACTION & MATCHING
        # ==========================================
        # Pad the bounding box slightly so we don't cut off edge features
        # ==========================================
        # ORB FEATURE EXTRACTION & MATCHING
        # ==========================================
        y1, y2 = max(0, y - 5), min(frame_gray.shape[0], y + h + 5)
        x1, x2 = max(0, x - 5), min(frame_gray.shape[1], x + w + 5)
        roi_gray = frame_gray[y1:y2, x1:x2]
        
        live_kp, live_des = orb.detectAndCompute(roi_gray, None)
        
        # Only process if we found enough texture in the crop
        if live_des is not None and len(live_kp) > MIN_MATCH_COUNT:
            best_orb_matches = 0
            
            for sym in symbol_templates:
                if sym["des"] is None:
                    continue
                    
                # 1. k-Nearest Neighbors Matching (find top 2 matches for each descriptor)
                # Need at least 2 descriptors in the template to do knnMatch(k=2)
                if len(sym["des"]) < 2: 
                    continue
                    
                matches = bf.knnMatch(sym["des"], live_des, k=2)
                
                # 2. Lowe's Ratio Test (Filter out ambiguous/weak matches)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        # 0.75 is the standard Lowe ratio. Lower = stricter.
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                
                # 3. Homography / RANSAC Geometry Check
                if len(good_matches) >= MIN_MATCH_COUNT:
                    # Extract the X/Y coordinates of the good matches
                    src_pts = np.float32([ sym["kp"][m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                    dst_pts = np.float32([ live_kp[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
                    
                    # Calculate the perspective transformation (RANSAC ignores outliers)
                    # The 5.0 is the RANSAC reprojection error threshold
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    # If M is not None, it means the points successfully form the shape!
                    if M is not None:
                        # Count how many points actually fit the strict geometric model (inliers)
                        matches_mask = mask.ravel().tolist()
                        valid_inliers = sum(matches_mask)
                        
                        if valid_inliers > best_orb_matches and valid_inliers >= MIN_MATCH_COUNT:
                            best_orb_matches = valid_inliers
                            best_match_name = sym["name"]

            # ==========================================
            # DRAWING & LABELING
            # ==========================================
            if best_match_name != "Unknown Symbol":
                box_color = (255, 255, 0) # Cyan for Symbols
                # Now we display the 'inliers' instead of raw matches
                label = f"{best_match_name} ({best_orb_matches} inliers)"
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x + (w - text_width) // 2
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, box_color, thickness)
                
                # Optional: Draw the live keypoints on the frame to visualize what ORB sees
                cv2.drawKeypoints(frame[y1:y2, x1:x2], live_kp, frame[y1:y2, x1:x2], color=(0,255,0), flags=0)

    # ==========================================
    # DISPLAY FEED
    # ==========================================
    cv2.imshow("Threshold (ROI Finder)", frame_threshold) 
    cv2.imshow("Live ORB Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
picam2.stop()       
cv2.destroyAllWindows()
