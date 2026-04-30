from picamera2 import Picamera2
import cv2 as cv
import numpy as np
import time

from config import *
from vision_utils import orb_match_symbol

# Initialize Camera & ORB
picam2 = Picamera2()
config_cam = picam2.create_preview_configuration(main={"size": (FRAME_WIDTH, FRAME_HEIGHT)})
picam2.configure(config_cam)
picam2.start()

orb = cv.ORB_create(nfeatures=500, nlevels=8, fastThreshold=17)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

# Load References
reference_data = []
for symbol_id, (img_files, threshold) in SYMBOL_DICT.items():
    refs = []
    for img_file in img_files:
        img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
        if img is not None:
            kp, des = orb.detectAndCompute(img, None)
            refs.append({"filename": img_file, "kp": kp, "des": des})
    reference_data.append({"id": symbol_id, "name": SYMBOL_NAME[symbol_id], "threshold": threshold, "refs": refs})
ref_by_id = {entry["id"]: entry for entry in reference_data}

print("SYMBOL DEBUGGER: Press 'c' to capture ORB stages, or 'q' to quit.")

while True:
    frame_rgb = picam2.capture_array()
    if frame_rgb.ndim == 3 and frame_rgb.shape[2] == 4: 
        frame_rgb = frame_rgb[:, :, :3]
        
    frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
    clean_bgr = frame_bgr.copy()
    gray_scene = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    
    blur = cv.GaussianBlur(frame_rgb, (3, 3), 0)
    HSV = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
    LAB = cv.cvtColor(blur, cv.COLOR_RGB2LAB)
    
    active_mask = None
    detected_color = "None"
    best_cnt = None
    symbol_label = ""
    roi_gray = None
    roi_with_keypoints = None
    
    all_candidates = []
    for colour_name, params in IMAGE_COLOUR_RANGES.items():
        src = HSV if params["space"] == "HSV" else LAB
        mask = cv.inRange(src, params["lower"], params["upper"])
        for cnt in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
            a = cv.contourArea(cnt)
            if a >= 1000: 
                all_candidates.append((a, cnt, colour_name, mask))
                
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    colour_to_ids = {"Yellow": [4], "Blue/Teal": [1, 2], "Purple": [1, 2], "Green": [0, 3]}

    if all_candidates:
        area, best_cnt, detected_color, active_mask = all_candidates[0]
        
        if detected_color in colour_to_ids:
            x, y, w, h = cv.boundingRect(best_cnt)
            pad = 15
            y1, y2 = max(0, y-pad), min(FRAME_HEIGHT, y+h+pad)
            x1, x2 = max(0, x-pad), min(FRAME_WIDTH, x+w+pad)
            
            roi_gray = gray_scene[y1:y2, x1:x2]
            kp_s, des_s = orb.detectAndCompute(roi_gray, None)
            
            if roi_gray is not None and roi_gray.size > 0:
                roi_with_keypoints = cv.drawKeypoints(roi_gray, kp_s, None, color=(0,255,0), flags=0)

            if des_s is not None:
                for sym_id in colour_to_ids[detected_color]:
                    entry = ref_by_id[sym_id]
                    matched, good_count = orb_match_symbol(bf, entry["refs"], des_s, entry["threshold"])
                    if matched:
                        symbol_label = entry["name"]
                        break
                        
            cv.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # --- NEW: Always draw the final text outcome at the top left ---
    status_text = f"Detected: {symbol_label}" if symbol_label else "Detected: None"
    cv.putText(frame_bgr, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)
    # ---------------------------------------------------------------

    cv.imshow("Symbol Detection Preview", frame_bgr)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('c'):
        ts = int(time.time() * 1000)
        cv.imwrite(f"{ts}_01_original.jpg", clean_bgr)
        if active_mask is not None:
            cv.imwrite(f"{ts}_02_mask_{detected_color}.jpg", active_mask)
            if roi_gray is not None:
                cv.imwrite(f"{ts}_03_roi_cropped.jpg", roi_gray)
            if roi_with_keypoints is not None:
                cv.imwrite(f"{ts}_04_roi_orb_keypoints.jpg", roi_with_keypoints)
        cv.imwrite(f"{ts}_05_final_output.jpg", frame_bgr)
        print(f"Saved symbol detection stages. {status_text}")
        break
    elif key == ord('q'):
        break

cv.destroyAllWindows()
picam2.stop()