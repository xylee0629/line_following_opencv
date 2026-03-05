import cv2
import numpy as np
from picamera2 import Picamera2
import time
from gpiozero import Motor, PWMOutputDevice
import math

# ==========================================
# ROBOT & CAMERA SETTINGS
# ==========================================
frame_width = 640
frame_height = 480
frame_centre = frame_width / 2

# Motor variables
frequency = 600
dutyCycle = 0.5
left_pwm = 0
right_pwm = 0

# PID variables
Kp = 0.004
Kd = 0.004
Ki = 0.0
last_error = 0
integral = 0 

# State Machine Flags
robot_active = False
# Added ALIGN_CARD to the states
robot_state = "FOLLOW_LINE"  # States: FOLLOW_LINE, ALIGN_CARD, IDENTIFY, WAIT_CLEAR
left_flag = 0
right_flag = 0

# Identification confirmation buffer
SQUARE_CONFIRM_FRAMES = 3
square_confirm_counter = 0

# WAIT_CLEAR timeout & cooldown
WAIT_CLEAR_TIMEOUT = 10.0       
WAIT_CLEAR_DEBOUNCE_FRAMES = 5  
wait_clear_start_time = 0.0
clear_frame_counter = 0
ignore_card_until = 0.0         

# GPIO pins
motorLeft  = Motor(19, 13)
ENA        = PWMOutputDevice(26, frequency=frequency)
motorRight = Motor(6, 5)
ENB        = PWMOutputDevice(22, frequency=frequency)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def stopMotors():
    motorLeft.stop()
    motorRight.stop()
    ENA.value = 0
    ENB.value = 0

def move(left_pwm_val, right_pwm_val):
    if left_pwm_val >= 0:
        motorLeft.forward()
        ENA.value = abs(left_pwm_val)
    else:
        motorLeft.backward()
        ENA.value = abs(left_pwm_val)

    if right_pwm_val >= 0:
        motorRight.forward()
        ENB.value = abs(right_pwm_val)
    else:
        motorRight.backward()
        ENB.value = abs(right_pwm_val)

def calculatePID(cx, error, current_dutyCycle):
    global last_error, integral 
    error = cx - frame_centre
    P = Kp * error
    integral += error
    I = Ki * integral
    derivative = error - last_error
    D = Kd * derivative
    control = P + I + D
    last_error = error
    left_p  = max(-1.0, min(1.0, current_dutyCycle + control))
    right_p = max(-1.0, min(1.0, current_dutyCycle - control))
    return left_p, right_p

def get_arrow_direction(arrow_contour):
    x, y, w, h = cv2.boundingRect(arrow_contour)
    box_center_x = x + (w / 2.0)
    box_center_y = y + (h / 2.0)
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0:
        return "UNKNOWN"
    com_x = M["m10"] / M["m00"]
    com_y = M["m01"] / M["m00"]
    dx = com_x - box_center_x
    dy = com_y - box_center_y
    if abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
    else:
        return "LEFT" if dx > 0 else "RIGHT"

def merge_nearby_contours(contours, frame_shape, proximity_threshold=80):
    if len(contours) == 0:
        return contours
    rects  = [cv2.boundingRect(c) for c in contours]
    parent = list(range(len(contours)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(len(rects)):
        xi, yi, wi, hi = rects[i]
        for j in range(i + 1, len(rects)):
            xj, yj, wj, hj = rects[j]
            if (xi - proximity_threshold < xj + wj and
                    xi + wi + proximity_threshold > xj and
                    yi - proximity_threshold < yj + hj and
                    yi + hi + proximity_threshold > yj):
                union(i, j)

    clusters = {}
    for i, cnt in enumerate(contours):
        clusters.setdefault(find(i), []).append(cnt)

    merged = []
    for group in clusters.values():
        if len(group) == 1:
            merged.append(group[0])
        else:
            all_pts = np.vstack(group)
            merged.append(cv2.convexHull(all_pts))
    return merged

MIN_SQUARE_AREA    = 3000
MAX_SQUARE_AREA    = 0.50
MIN_CARD_SATURATION = 40

def find_best_square_contour(contours, frame_area, frame_hsv):
    best_cnt  = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SQUARE_AREA or area > frame_area * MAX_SQUARE_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0
        if not (0.70 <= aspect_ratio <= 1.43):
            continue

        roi_sat = frame_hsv[y:y + h, x:x + w, 1]
        mean_sat = float(roi_sat.mean())
        if mean_sat < MIN_CARD_SATURATION:
            continue

        if area > best_area:
            best_area = area
            best_cnt  = cnt

    return best_cnt

# ==========================================
# LOAD TEMPLATES (GEOMETRY & ORB)
# ==========================================
reference_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/shapes.png',
     "names": ["Star", "Diamond", "Cross", "Trapezoid", "3/4 Semicircle", "Semicircle", "Octagon"]},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/arrow.png',
     "names": ["Arrow"]}
]

geo_templates = []
print("\n--- LEARNING REFERENCE SHAPES ---")
for ref_data in reference_files:
    ref_img = cv2.imread(ref_data["filepath"])
    if ref_img is None: continue
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    _, ref_saturation, _ = cv2.split(ref_hsv)
    ref_blur = cv2.GaussianBlur(ref_saturation, (15, 15), 0)
    _, ref_threshold = cv2.threshold(ref_blur, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25), np.uint8)
    ref_threshold = cv2.morphologyEx(ref_threshold, cv2.MORPH_CLOSE, kernel)
    ref_contours, _ = cv2.findContours(ref_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = sorted(
        [c for c in ref_contours if cv2.contourArea(c) > 100],
        key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0])
    )
    for i, cnt in enumerate(valid_contours):
        name = ref_data["names"][i] if i < len(ref_data["names"]) else f"Extra {i}"
        hull = cv2.convexHull(cnt)
        solidity = cv2.contourArea(cnt) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        rect = cv2.minAreaRect(cnt)
        rw, rh = rect[1]
        aspect_ratio = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
        extent = cv2.contourArea(cnt) / (rw * rh) if (rw * rh) > 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * cv2.contourArea(cnt)) / (perimeter ** 2) if perimeter > 0 else 0
        geo_templates.append({
            "name": name, "contour": cnt,
            "solidity": solidity, "aspect_ratio": aspect_ratio,
            "circularity": circularity, "extent": extent
        })
        print(f"  -> Learned: {name}")

orb = cv2.ORB_create(nfeatures=800, fastThreshold=10, edgeThreshold=10)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
MIN_MATCH_COUNT = 14

symbol_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/button.jpg',      "name": "Button"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/fingerprint.jpg', "name": "Fingerprint"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/hazard.jpg',      "name": "Hazard"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/qr.jpg',          "name": "QR Code"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/recycle.jpg',     "name": "Recycle"}
]

orb_templates = []
print("\n--- LEARNING REFERENCE SYMBOLS ---")
for sym_data in symbol_files:
    sym_img = cv2.imread(sym_data["filepath"], cv2.IMREAD_GRAYSCALE)
    if sym_img is None: continue
    kp, des = orb.detectAndCompute(sym_img, None)
    if des is not None:
        orb_templates.append({"name": sym_data["name"], "kp": kp, "des": des})
        print(f"  -> Learned: {sym_data['name']} ({len(kp)} keypoints)")

# ==========================================
# CAMERA START
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"format": 'XRGB8888', "size": (frame_width, frame_height)}
))
picam2.start()
print("\nReady! Press 's' to start moving. Press 'q' to quit.")
time.sleep(1)

frame_area = frame_width * frame_height

# ==========================================
# MAIN LOOP
# ==========================================
try:
    while True:
        frame     = picam2.capture_array()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Shared Preprocessing
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, frame_saturation, _ = cv2.split(frame_hsv)
        frame_blur_sym = cv2.GaussianBlur(frame_saturation, (15, 15), 0)
        ret_sym, frame_thresh_sym = cv2.threshold(frame_blur_sym, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((25, 25), np.uint8)
        frame_thresh_sym = cv2.morphologyEx(frame_thresh_sym, cv2.MORPH_CLOSE, kernel)
        if ret_sym < 40:
            frame_thresh_sym = np.zeros_like(frame_thresh_sym)

        raw_contours, _ = cv2.findContours(frame_thresh_sym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_contours = merge_nearby_contours(raw_contours, frame.shape, proximity_threshold=20)

        square_contour_found = find_best_square_contour(frame_contours, frame_area, frame_hsv)

        # ------------------------------------------------------------------
        # STATE MACHINE
        # ------------------------------------------------------------------
        if robot_state == "FOLLOW_LINE":
            
            if time.time() < ignore_card_until:
                square_contour_found = None

            if square_contour_found is not None:
                square_confirm_counter += 1
            else:
                square_confirm_counter = 0

            if square_confirm_counter >= SQUARE_CONFIRM_FRAMES:
                square_confirm_counter = 0
                move(0, 0)
                # Redirecting from FOLLOW_LINE to ALIGN_CARD
                robot_state = "ALIGN_CARD"
                print(f"\n[!] Square confirmed — aligning card in frame...")
                continue

            # Standard line following
            blur_line  = cv2.GaussianBlur(frame_gray, (5, 5), 0)
            _, thresh_line = cv2.threshold(blur_line, 40, 255, cv2.THRESH_BINARY_INV)

            if square_contour_found is not None:
                cx_card, cy_card, cw_card, ch_card = cv2.boundingRect(square_contour_found)
                thresh_line[cy_card:cy_card + ch_card, cx_card:cx_card + cw_card] = 0

            line_contours, _ = cv2.findContours(thresh_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(line_contours) > 0:
                largest_line = max(line_contours, key=cv2.contourArea)
                M = cv2.moments(largest_line)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    left_flag  = 1 if cx <= frame_centre else 0
                    right_flag = 0 if cx <= frame_centre else 1
            else:
                if left_flag == 1:       cx = 0
                elif right_flag == 1:    cx = frame_width
                else:                    cx = int(frame_centre)

            error = cx - frame_centre
            if error == 0:
                left_pwm, right_pwm = dutyCycle, dutyCycle
            else:
                left_pwm, right_pwm = calculatePID(cx, error, dutyCycle)

            if robot_active:
                move(left_pwm, right_pwm)
            else:
                move(0, 0)
                cv2.putText(frame, "STANDBY - Press 's' to start",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        # NEW STATE: Visually servo the robot to center the card
        elif robot_state == "ALIGN_CARD":
            if square_contour_found is None:
                print("[!] Lost card during alignment. Resuming line follow.")
                robot_state = "FOLLOW_LINE"
            else:
                x, y, w, h = cv2.boundingRect(square_contour_found)
                
                # Define safe margins (20 pixels from top and bottom)
                top_margin = 20
                bottom_margin = frame_height - 20
                
                # Slower motor speed for precise alignment
                align_speed = 0.35 

                if y + h > bottom_margin:
                    # Card is cut off at the bottom -> Drove too far, move backward
                    move(-align_speed, -align_speed)
                    cv2.putText(frame, "ALIGNING: Reversing...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    
                elif y < top_margin:
                    # Card is cut off at the top -> Backed up too far, creep forward
                    move(align_speed, align_speed)
                    cv2.putText(frame, "ALIGNING: Creeping Forward...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    
                else:
                    # Card is fully floating inside the safe frame!
                    move(0, 0)
                    print("[!] Card fully in frame. Commencing identification...")
                    robot_state = "IDENTIFY"


        elif robot_state == "IDENTIFY":
            if square_contour_found is None:
                print("[!] Square lost before identification. Resuming line follow.")
                robot_state = "FOLLOW_LINE"
            else:
                cnt  = square_contour_found
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)

                # --- Pipeline A: Geometry ---
                live_hull      = cv2.convexHull(cnt)
                live_hull_area = cv2.contourArea(live_hull)
                live_solidity  = area / live_hull_area if live_hull_area > 0 else 0
                rect_w, rect_h = cv2.minAreaRect(cnt)[1]
                live_aspect_ratio = (max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0)
                live_extent    = area / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
                live_perimeter = cv2.arcLength(cnt, True)
                live_circularity = ((4 * np.pi * area) / (live_perimeter ** 2) if live_perimeter > 0 else 0)

                geo_best_name  = "Unknown"
                geo_best_score = 1.5

                for template in geo_templates:
                    hu_score    = cv2.matchShapes(template["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                    total_score = (hu_score
                                   + abs(template["solidity"]     - live_solidity)     * 2.0
                                   + abs(template["aspect_ratio"] - live_aspect_ratio) * 0.5
                                   + abs(template["circularity"]  - live_circularity)
                                   + abs(template["extent"]       - live_extent)       * 3.0)
                    if total_score < geo_best_score:
                        geo_best_score = total_score
                        geo_best_name  = template["name"]

                # --- Pipeline B: ORB ---
                y1 = max(0, y - 5);          y2 = min(frame_gray.shape[0], y + h + 5)
                x1 = max(0, x - 5);          x2 = min(frame_gray.shape[1], x + w + 5)
                roi_gray = frame_gray[y1:y2, x1:x2]
                live_kp, live_des = orb.detectAndCompute(roi_gray, None)

                orb_best_name    = "Unknown Symbol"
                orb_best_inliers = 0

                if live_des is not None and len(live_kp) > 45:
                    for sym in orb_templates:
                        if sym["des"] is None: continue
                        
                        matches = bf.match(sym["des"], live_des)
                        good_matches = [m for m in matches if m.distance < 60]
                        
                        if len(good_matches) >= MIN_MATCH_COUNT:
                            src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            
                            if M_hom is not None:
                                valid_inliers = int(mask.ravel().sum())
                                if valid_inliers > orb_best_inliers and valid_inliers >= MIN_MATCH_COUNT:
                                    orb_best_inliers = valid_inliers
                                    orb_best_name    = sym["name"]

                # Final Decision
                if orb_best_name != "Unknown Symbol":
                    print(f"-> IDENTIFIED SYMBOL: {orb_best_name} (Inliers: {orb_best_inliers})")
                elif geo_best_name != "Unknown":
                    display_name = (f"Arrow {get_arrow_direction(cnt)}" if "Arrow" in geo_best_name else geo_best_name)
                    print(f"-> IDENTIFIED SHAPE: {display_name} (Score: {geo_best_score:.2f})")
                else:
                    print("-> UNKNOWN SQUARE OBJECT.")

                wait_clear_start_time = time.time()
                clear_frame_counter   = 0
                robot_state = "WAIT_CLEAR"


        elif robot_state == "WAIT_CLEAR":
            move(0, 0)
            elapsed = time.time() - wait_clear_start_time

            cv2.putText(frame, "OBSTACLE DETECTED - PLEASE REMOVE",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Timeout in {max(0, WAIT_CLEAR_TIMEOUT - elapsed):.1f}s",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            if square_contour_found is None:
                clear_frame_counter += 1
            else:
                clear_frame_counter = 0

            path_clear   = clear_frame_counter >= WAIT_CLEAR_DEBOUNCE_FRAMES
            timed_out    = elapsed >= WAIT_CLEAR_TIMEOUT

            if path_clear or timed_out:
                reason = "Path clear" if path_clear else "Timeout reached"
                print(f"[!] {reason}. Resuming line follow.")
                
                if timed_out:
                    ignore_card_until = time.time() + 3.0
                    
                clear_frame_counter = 0
                robot_state = "FOLLOW_LINE"

        # ------------------------------------------------------------------
        # DISPLAY
        # ------------------------------------------------------------------
        blur_display = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        _, thresh_display = cv2.threshold(blur_display, 40, 255, cv2.THRESH_BINARY_INV)
        if square_contour_found is not None:
            dx, dy, dw, dh = cv2.boundingRect(square_contour_found)
            thresh_display[dy:dy + dh, dx:dx + dw] = 0 
        display_line_contours, _ = cv2.findContours(thresh_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(display_line_contours) > 0:
            largest_display_line = max(display_line_contours, key=cv2.contourArea)
            lx, ly, lw, lh = cv2.boundingRect(largest_display_line)
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), (255, 100, 0), 2)
            cv2.putText(frame, "Line", (lx, ly - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 2)
            lM = cv2.moments(largest_display_line)
            if lM["m00"] != 0:
                lcx = int(lM["m10"] / lM["m00"])
                lcy = int(lM["m01"] / lM["m00"])
                cv2.circle(frame, (lcx, lcy), 6, (255, 100, 0), -1)

        if square_contour_found is not None:
            sx, sy, sw, sh = cv2.boundingRect(square_contour_found)
            
            # Dynamic Box Coloring Based on State
            if robot_state == "FOLLOW_LINE":
                card_colour = (0, 220, 255)
                card_label  = f"Card? ({square_confirm_counter}/{SQUARE_CONFIRM_FRAMES})"
            elif robot_state == "ALIGN_CARD":
                card_colour = (255, 100, 255) 
                card_label  = "Aligning..."
            elif robot_state == "IDENTIFY":
                card_colour = (0, 140, 255)
                card_label  = "Identifying..."
            elif robot_state == "WAIT_CLEAR":
                card_colour = (0, 0, 255)
                card_label  = "Remove card"
            else:
                card_colour = (0, 220, 255)
                card_label  = "Card"

            cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), card_colour, 2)
            cv2.putText(frame, card_label, (sx, sy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, card_colour, 2)

        cv2.putText(frame, f"State: {robot_state}", (10, frame_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        
        cv2.imshow("Robot View", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            robot_active = True
            print("Motors Activated.")

except KeyboardInterrupt:
    print("Program stopped by user")

finally:
    stopMotors()
    ENA.close()
    ENB.close()
    picam2.stop()
    cv2.destroyAllWindows()