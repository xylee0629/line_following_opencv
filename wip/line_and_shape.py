import cv2
import numpy as np
from picamera2 import Picamera2
import time
import math
from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice, Servo

# ==========================================
# CAMERA CONFIG
# ==========================================
frame_width  = 640
frame_height = 460
frame_centre = frame_width / 2

# ==========================================
# MOTOR / ENCODER CONFIG
# ==========================================
frequency       = 600
dutyCycle       = 0.4
diameter        = 8.5
encoder_slots   = 20
circumference   = math.pi * diameter
distance_per_tick = circumference / encoder_slots

encoderState = {"left": 0, "right": 0}

motorLeft  = Motor(19, 13)
ENA        = PWMOutputDevice(26, frequency=frequency)
motorRight = Motor(6, 5)
ENB        = PWMOutputDevice(22, frequency=frequency)
encoderLeft  = DigitalInputDevice(27, pull_up=False)
encoderRight = DigitalInputDevice(21, pull_up=False)
servo = Servo(20)

# ==========================================
# PID CONFIG
# ==========================================
Kp         = 0.004
Kd         = 0.004
Ki         = 0.0
last_error = 0

# ==========================================
# MOTOR HELPERS
# ==========================================
def stopMotors():
    motorLeft.stop()
    motorRight.stop()
    time.sleep(0.5)

def move(left_pwm, right_pwm):
    if left_pwm >= 0:
        motorLeft.forward()
    else:
        motorLeft.backward()
    ENA.value = abs(left_pwm)

    if right_pwm >= 0:
        motorRight.forward()
    else:
        motorRight.backward()
    ENB.value = abs(right_pwm)

def calculatePID(cx, dutyCycle):
    global last_error
    error      = cx - frame_centre
    P          = Kp * error
    D          = Kd * (error - last_error)
    control    = P + D
    last_error = error
    left_pwm   = max(-1.0, min(1.0, dutyCycle + control))
    right_pwm  = max(-1.0, min(1.0, dutyCycle - control))
    return left_pwm, right_pwm

# ==========================================
# SYMBOL DETECTOR HELPERS
# ==========================================
def get_arrow_direction(arrow_contour):
    x, y, w, h = cv2.boundingRect(arrow_contour)
    box_cx = x + w / 2.0
    box_cy = y + h / 2.0
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0:
        return "UNKNOWN"
    com_x = M["m10"] / M["m00"]
    com_y = M["m01"] / M["m00"]
    dx = com_x - box_cx
    dy = com_y - box_cy
    if abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
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

# ==========================================
# LOAD GEOMETRY TEMPLATES
# ==========================================
reference_files = [
    {
        "filepath": '/home/raspberrypi/line_following_opencv/images/shapes.png',
        "names": ["Star", "Diamond", "Cross", "Trapezoid", "3/4 Semicircle", "Semicircle", "Octagon"]
    },
    {
        "filepath": '/home/raspberrypi/line_following_opencv/images/arrow.png',
        "names": ["Arrow"]
    }
]

geo_templates = []
print("\n--- LEARNING GEOMETRY TEMPLATES ---")
for ref_data in reference_files:
    ref_img = cv2.imread(ref_data["filepath"])
    if ref_img is None:
        print(f"ERROR: Could not load {ref_data['filepath']}")
        continue

    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    _, ref_sat, _ = cv2.split(ref_hsv)
    ref_blur = cv2.GaussianBlur(ref_sat, (15, 15), 0)
    _, ref_thresh = cv2.threshold(ref_blur, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((25, 25), np.uint8)
    ref_thresh = cv2.morphologyEx(ref_thresh, cv2.MORPH_CLOSE, kernel)

    ref_contours, _ = cv2.findContours(ref_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in ref_contours if 100 < cv2.contourArea(c) < (ref_img.shape[0] * ref_img.shape[1] * 0.9)]
    valid = sorted(valid, key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0]))

    for i, cnt in enumerate(valid):
        name = ref_data["names"][i] if i < len(ref_data["names"]) else f"Shape {i}"
        hull_area  = cv2.contourArea(cv2.convexHull(cnt))
        area       = cv2.contourArea(cnt)
        solidity   = area / hull_area if hull_area > 0 else 0
        rect       = cv2.minAreaRect(cnt)
        w, h       = rect[1]
        ar         = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        extent     = area / (w * h) if (w * h) > 0 else 0
        perim      = cv2.arcLength(cnt, True)
        circ       = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
        geo_templates.append({"name": name, "contour": cnt,
                               "solidity": solidity, "aspect_ratio": ar,
                               "circularity": circ, "extent": extent})
        print(f"  -> Learned: {name}")

print(f"Loaded {len(geo_templates)} geometry templates.")

# ==========================================
# LOAD ORB TEMPLATES
# ==========================================
orb = cv2.ORB_create()
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
MIN_MATCH_COUNT = 6

symbol_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/button.jpg',      "name": "Button"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/fingerprint.jpg', "name": "Fingerprint"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/hazard.jpg',      "name": "Hazard"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/qr.jpg',          "name": "QR Code"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/recycle.jpg',     "name": "Recycle"}
]

orb_templates = []
print("\n--- LEARNING ORB TEMPLATES ---")
for sym_data in symbol_files:
    img = cv2.imread(sym_data["filepath"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"ERROR: Could not load {sym_data['filepath']}")
        continue
    kp, des = orb.detectAndCompute(img, None)
    if des is not None:
        orb_templates.append({"name": sym_data["name"], "kp": kp, "des": des})
        print(f"  -> Learned: {sym_data['name']} ({len(kp)} keypoints)")

print(f"Loaded {len(orb_templates)} ORB templates.")

# ==========================================
# CAMERA INIT
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"format": 'XRGB8888', "size": (frame_width, frame_height)}))
picam2.start()
time.sleep(1)

# ==========================================
# STATE
# ==========================================
robot_active    = False
left_flag       = 0
right_flag      = 0
left_pwm        = 0
right_pwm       = 0
symbol_streak   = 0     # consecutive frames a symbol has been seen
STREAK_REQUIRED = 1    # frames needed before the car actually stops (~8 frames)
stop_until      = 0.0   # time.time() value until which the car should remain stopped
STOP_DURATION   = 10.0  # seconds to stop when a symbol is confirmed

print("\nReady. Press 's' to start, 'q' to quit.")

try:
    while True:
        frame = picam2.capture_array()

        # ---- SHARED ----
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, frame_sat, _ = cv2.split(frame_hsv)

        # ==========================================
        # PIPELINE A: LINE FOLLOWING (grayscale)
        # ==========================================
        blur_line = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        _, thresh_line = cv2.threshold(blur_line, 40, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = int(frame_centre)   # default: centre
        cy = frame_height // 2
        largest_contour = None

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                left_flag  = 1 if cx <= frame_centre else 0
                right_flag = 0 if cx <= frame_centre else 1
        else:
            if left_flag == 1:
                cx = 0
            elif right_flag == 1:
                cx = frame_width

        error = cx - frame_centre
        if error == 0:
            left_pwm = right_pwm = dutyCycle
        else:
            left_pwm, right_pwm = calculatePID(cx, dutyCycle)

        # ==========================================
        # PIPELINE B: SYMBOL DETECTION (saturation)
        # ==========================================
        blur_sym = cv2.GaussianBlur(frame_sat, (15, 15), 0)
        ret_sym, thresh_sym = cv2.threshold(blur_sym, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((30, 30), np.uint8)
        thresh_sym = cv2.morphologyEx(thresh_sym, cv2.MORPH_CLOSE, kernel)

        if ret_sym < 40:
            thresh_sym = np.zeros_like(thresh_sym)

        raw_contours, _ = cv2.findContours(thresh_sym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sym_contours = merge_nearby_contours(raw_contours, frame.shape, proximity_threshold=60)

        detected_symbol = None   # will hold the label string if anything is found

        for cnt in sym_contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # --- Geometry match ---
            live_hull_area = cv2.contourArea(cv2.convexHull(cnt))
            live_solidity  = area / live_hull_area if live_hull_area > 0 else 0
            live_rect      = cv2.minAreaRect(cnt)
            rw, rh         = live_rect[1]
            live_ar        = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 0
            live_extent    = area / (rw * rh) if (rw * rh) > 0 else 0
            live_perim     = cv2.arcLength(cnt, True)
            live_circ      = (4 * np.pi * area) / (live_perim ** 2) if live_perim > 0 else 0

            geo_best_name  = "Unknown"
            geo_best_score = 1.5

            for template in geo_templates:
                hu_score    = cv2.matchShapes(template["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                sol_diff    = abs(template["solidity"]      - live_solidity)
                ar_diff     = abs(template["aspect_ratio"]  - live_ar) * 0.5
                circ_diff   = abs(template["circularity"]   - live_circ)
                extent_diff = abs(template["extent"]        - live_extent)
                score = hu_score + (sol_diff * 2.0) + ar_diff + circ_diff + (extent_diff * 3.0)
                if score < geo_best_score:
                    geo_best_score = score
                    geo_best_name  = template["name"]

            # --- ORB match ---
            y1 = max(0, y - 5);  y2 = min(frame_gray.shape[0], y + h + 5)
            x1 = max(0, x - 5);  x2 = min(frame_gray.shape[1], x + w + 5)
            roi_gray = frame_gray[y1:y2, x1:x2]

            live_kp, live_des = orb.detectAndCompute(roi_gray, None)

            orb_best_name    = "Unknown Symbol"
            orb_best_inliers = 0

            if live_des is not None and len(live_kp) > MIN_MATCH_COUNT:
                for sym in orb_templates:
                    if sym["des"] is None or len(sym["des"]) < 2:
                        continue
                    matches = bf.knnMatch(sym["des"], live_des, k=2)
                    good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
                    if len(good) >= MIN_MATCH_COUNT:
                        src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([live_kp[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
                        M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M_hom is not None:
                            inliers = int(mask.ravel().sum())
                            if inliers > orb_best_inliers and inliers >= MIN_MATCH_COUNT:
                                orb_best_inliers = inliers
                                orb_best_name    = sym["name"]

            # --- Draw & record ---
            if orb_best_name != "Unknown Symbol":
                label     = f"{orb_best_name} ({orb_best_inliers} inliers)"
                box_color = (255, 255, 0)
                detected_symbol = orb_best_name
            elif geo_best_name != "Unknown":
                if "Arrow" in geo_best_name:
                    direction = get_arrow_direction(cnt)
                    geo_best_name = f"Arrow {direction}"
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 255, 0)
                label = f"{geo_best_name} ({geo_best_score:.2f})"
                detected_symbol = geo_best_name
            else:
                continue   # nothing recognised — skip drawing

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            text_x = x + (w - tw) // 2
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, label, (text_x, text_y), font, 0.6, box_color, 2)

        # ==========================================
        # SYMBOL STREAK COUNTER
        # Require STREAK_REQUIRED consecutive frames before confirming a symbol.
        # This prevents a single noisy frame from stopping the car.
        # ==========================================
        if detected_symbol:
            symbol_streak += 1
            if symbol_streak == STREAK_REQUIRED:
                stop_until = time.time() + STOP_DURATION
                print(f"[SYMBOL CONFIRMED] {detected_symbol} — stopping for {STOP_DURATION}s.")
        else:
            symbol_streak = 0   # reset the moment symbol leaves frame

        currently_stopped = time.time() < stop_until

        # ==========================================
        # LINE FOLLOWING OVERLAY & MOTOR OUTPUT
        # ==========================================
        if largest_contour is not None:
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

        if not robot_active:
            move(0, 0)
            cv2.putText(frame, "STANDBY - Press 's' to start",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif currently_stopped:
            move(0, 0)
            remaining = int(stop_until - time.time()) + 1
            cv2.putText(frame, f"STOPPED: {detected_symbol or 'Symbol'} ({remaining}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            move(left_pwm, right_pwm)

        cv2.imshow("Combined: Line Follow + Symbol Detect", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            robot_active = True
            print("Motors started.")

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    stopMotors()
    ENA.close()
    ENB.close()
    picam2.stop()
    cv2.destroyAllWindows()