import cv2
import numpy as np
import time
import math
import threading
import sys
from picamera2 import Picamera2
from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice, Servo

# ==========================================
# 1. HARDWARE & PID SETUP
# ==========================================
frame_width = 640
frame_height = 480 
frame_centre = frame_width / 2

# Motor variables 
frequency = 600
dutyCycle = 0.35
Kp = 0.003
Kd = 0.008
Ki = 0.0
last_error = 0

# GPIO pins 
motorLeft = Motor(19, 13)
ENA = PWMOutputDevice(26, frequency=frequency)
motorRight = Motor(6, 5)
ENB = PWMOutputDevice(22, frequency=frequency)

# ==========================================
# 2. VISION HELPERS & TEMPLATES
# ==========================================
def get_arrow_direction(arrow_contour):
    x, y, w, h = cv2.boundingRect(arrow_contour)
    box_center_x, box_center_y = x + (w / 2.0), y + (h / 2.0)
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0: return "UNKNOWN"
    dx = (M["m10"] / M["m00"]) - box_center_x
    dy = (M["m01"] / M["m00"]) - box_center_y
    if abs(dy) > abs(dx): return "DOWN" if dy > 0 else "UP"
    else: return "LEFT" if dx > 0 else "RIGHT"

def merge_nearby_contours(contours, proximity_threshold=60):
    if not contours: return []
    rects = [cv2.boundingRect(c) for c in contours]
    parent = list(range(len(contours)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    for i, (xi, yi, wi, hi) in enumerate(rects):
        for j in range(i + 1, len(rects)):
            xj, yj, wj, hj = rects[j]
            if (xi - proximity_threshold < xj + wj and xi + wi + proximity_threshold > xj and
                yi - proximity_threshold < yj + hj and yi + hi + proximity_threshold > yj):
                union(i, j)
    clusters = {}
    for i, cnt in enumerate(contours):
        clusters.setdefault(find(i), []).append(cnt)
    return [cv2.convexHull(np.vstack(g)) if len(g) > 1 else g[0] for g in clusters.values()]

# Load Arrow Templates
arrow_templates = []
arrow_img = cv2.imread('/home/raspberrypi/line_following_opencv/images/arrow.png')
if arrow_img is not None:
    _, sat, _ = cv2.split(cv2.cvtColor(arrow_img, cv2.COLOR_BGR2HSV))
    _, thresh = cv2.threshold(cv2.GaussianBlur(sat, (15, 15), 0), 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            arrow_templates.append({
                "contour": cnt,
                "solidity": cv2.contourArea(cnt) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0,
                "aspect_ratio": max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                "extent": cv2.contourArea(cnt) / (w * h) if (w * h) > 0 else 0,
                "circularity": (4 * np.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
            })

# Load ORB Templates
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
MIN_MATCH_COUNT = 6
symbol_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/button.jpg', "name": "Button"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/fingerprint.jpg', "name": "Fingerprint"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/hazard.jpg', "name": "Hazard"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/qr.jpg', "name": "QR Code"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/recycle.jpg', "name": "Recycle"}
]
orb_templates = []
for sym in symbol_files:
    img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None: orb_templates.append({"name": sym["name"], "kp": kp, "des": des})

# ==========================================
# 3. MOVEMENT & CONTROL
# ==========================================
def stopMotors():
    motorLeft.stop()
    motorRight.stop()
    ENA.value = 0
    ENB.value = 0

def move(left_pwm, right_pwm):
    if left_pwm >= 0: motorLeft.forward()
    else: motorLeft.backward()
    ENA.value = abs(left_pwm)
        
    if right_pwm >= 0: motorRight.forward()
    else: motorRight.backward()
    ENB.value = abs(right_pwm)

def calculatePID(cx):
    global last_error
    error = cx - frame_centre
    P = Kp * error
    D = Kd * (error - last_error)
    control = P + D # Skipping Ki since it's 0
    last_error = error
    
    left_pwm = max(-1.0, min(1.0, dutyCycle + control))
    right_pwm = max(-1.0, min(1.0, dutyCycle - control))
    return left_pwm, right_pwm

# ==========================================
# 4. BACKGROUND THREAD (TERMINAL LISTENER)
# ==========================================
running = True
robot_active = False

def terminal_listener():
    """Runs in the background to listen for terminal commands."""
    global robot_active, running
    
    print("\n--- Terminal Control Active ---")
    print("Type 's' + Enter to toggle Start/Pause")
    print("Type 'q' + Enter to Quit\n")
    
    while running:
        cmd = sys.stdin.readline().strip().lower()
        if cmd == 's':
            robot_active = not robot_active
            state = "STARTED" if robot_active else "PAUSED"
            print(f"\n[COMMAND] Motors {state}!")
        elif cmd == 'q':
            print("\n[COMMAND] Shutting down...")
            running = False

# ==========================================
# 5. MAIN LOOP
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'BGR888', "size": (frame_width, frame_height)}))
picam2.start()
time.sleep(1)

left_flag, right_flag = 0, 0
frame_count = 0

# Start the background listener thread
input_thread = threading.Thread(target=terminal_listener, daemon=True)
input_thread.start()

try:
    while running:
        # 1. Capture the raw 4-channel image
        frame = picam2.capture_array()
        frame_count += 1
        
        # --- ROI SPLIT ---
        top_roi = frame[0:400, :]
        bottom_roi = frame[240:480, :]
        
        # ==========================================
        # TASK A: LINE FOLLOWING (Every Frame)
        # ==========================================
        gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                if cx <= frame_centre: left_flag, right_flag = 1, 0
                elif cx > frame_centre: left_flag, right_flag = 0, 1
        else:
            if left_flag == 1: cx = 0
            elif right_flag == 1: cx = frame_width
            else: cx = frame_centre
                
        if robot_active:
            left_pwm, right_pwm = calculatePID(cx)
            move(left_pwm, right_pwm)
        else:
            move(0, 0)

        # ==========================================
        # TASK B: SYMBOL DETECTION (Every 5th Frame)
        # ==========================================
        if frame_count % 3 == 0:
            top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
            _, top_sat, _ = cv2.split(cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV))
            ret_sym, top_thresh = cv2.threshold(cv2.GaussianBlur(top_sat, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if ret_sym >= 40:
                top_thresh = cv2.morphologyEx(top_thresh, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
                sym_contours, _ = cv2.findContours(top_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sym_contours = merge_nearby_contours(sym_contours, proximity_threshold=60)

                for cnt in sym_contours:
                    area = cv2.contourArea(cnt)
                    
                    # 1. Calculate the bounding box FIRST so y and h exist
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    aspect_ratio = float(w) / h if h != 0 else 0
                    
                    # 1. Noise Filter: Ignore tiny blips
                    if area < 500: 
                        continue
                    
                    # 1. ORB Check
                    live_kp, live_des = orb.detectAndCompute(top_gray[y:y+h, x:x+w], None)
                    orb_matched = False
                    if live_des is not None and len(live_kp) > MIN_MATCH_COUNT:
                        best_inliers, best_name = 0, ""
                        for sym in orb_templates:
                            matches = bf.knnMatch(sym["des"], live_des, k=2)
                            good = [m for m, n in matches if m.distance < 0.75 * n.distance] if len(matches[0]) == 2 else []
                            if len(good) >= MIN_MATCH_COUNT:
                                src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                                dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                                M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                                if M_hom is not None:
                                    inliers = mask.sum()
                                    if inliers > best_inliers and inliers >= MIN_MATCH_COUNT:
                                        best_inliers, best_name = inliers, sym["name"]
                        if best_inliers > 0:
                            print(f"[SIGN DETECTED] -> {best_name}")
                            orb_matched = True

                    # 2. Arrow Check (If no ORB match)
                    if not orb_matched and arrow_templates:
                        live_hull = cv2.convexHull(cnt)
                        live_rect = cv2.minAreaRect(cnt)
                        rect_w, rect_h = live_rect[1]
                        live_sol = area / cv2.contourArea(live_hull) if cv2.contourArea(live_hull) > 0 else 0
                        live_ar = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
                        live_ext = area / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
                        live_circ = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0

                        best_score = 1.5
                        for temp in arrow_templates:
                            score = cv2.matchShapes(temp["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                            score += abs(temp["solidity"] - live_sol) * 2.0
                            score += abs(temp["aspect_ratio"] - live_ar) * 0.5
                            score += abs(temp["circularity"] - live_circ)
                            score += abs(temp["extent"] - live_ext) * 3.0
                            if score < best_score: best_score = score

                        if best_score < 1.0:
                            print(f"[ARROW DETECTED] -> Direction: {get_arrow_direction(cnt)}")

        # Optional: Keep imshow for debugging, but we removed waitKey.
        # If you run this over SSH without X11 forwarding, comment the line below out.
        cv2.imshow("Original Feed", frame)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nProgram stopped by Ctrl+C") 

finally:
    running = False # Ensures the thread also shuts down cleanly
    stopMotors()
    ENA.close()
    ENB.close()
    picam2.stop()
    cv2.destroyAllWindows()
