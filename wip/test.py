import cv2
import numpy as np
from picamera2 import Picamera2
import time
import math
from collections import deque, Counter
from gpiozero import Motor, DigitalInputDevice, PWMOutputDevice, Servo

# ==========================================
# CAMERA CONFIG
# ==========================================
frame_width  = 640
frame_height = 360
frame_centre = frame_width / 2

# ==========================================
# ROI BOUNDARIES
# Upper ROI  → symbol / shape detection only
# Lower ROI  → line following only
# ==========================================
UPPER_Y1 = 0
UPPER_Y2 = 200          # symbol detector sees top 200 rows

LOWER_Y1 = 180          # slight overlap keeps transition smooth
LOWER_Y2 = frame_height
LOWER_X1 = 80
LOWER_X2  = 560

LOWER_CENTRE = (LOWER_X2 - LOWER_X1) // 2   # PID reference inside lower ROI

# ==========================================
# STATE MACHINE TUNING
# ==========================================
LABEL_HISTORY_LEN   = 5     # rolling window of recent detections
MIN_CONFIRM_COUNT   = 3     # votes needed in window to confirm a symbol
STOP_DURATION       = 10.0  # seconds the car stops after confirming
COOLDOWN_AFTER_HOLD = 2.0   # seconds before detection re-arms after resuming

# ==========================================
# MOTOR / ENCODER CONFIG
# ==========================================
frequency         = 600
dutyCycle         = 0.5
diameter          = 8.5
encoder_slots     = 20
circumference     = math.pi * diameter
distance_per_tick = circumference / encoder_slots

encoderState = {"left": 0, "right": 0}

motorLeft    = Motor(19, 13)
ENA          = PWMOutputDevice(26, frequency=frequency)
motorRight   = Motor(6, 5)
ENB          = PWMOutputDevice(22, frequency=frequency)
encoderLeft  = DigitalInputDevice(27, pull_up=False)
encoderRight = DigitalInputDevice(21, pull_up=False)
servo        = Servo(20)

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
    error      = cx - LOWER_CENTRE     # error relative to lower ROI centre
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
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0:
        return "UNKNOWN"
    com_x = M["m10"] / M["m00"]
    com_y = M["m01"] / M["m00"]
    dx = com_x - (x + w / 2.0)
    dy = com_y - (y + h / 2.0)
    if abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
    return "LEFT" if dx > 0 else "RIGHT"

def merge_nearby_contours(contours, proximity_threshold=80):
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
            merged.append(cv2.convexHull(np.vstack(group)))
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
        name      = ref_data["names"][i] if i < len(ref_data["names"]) else f"Shape {i}"
        area      = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity  = area / hull_area if hull_area > 0 else 0
        rect      = cv2.minAreaRect(cnt)
        w, h      = rect[1]
        ar        = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        extent    = area / (w * h) if (w * h) > 0 else 0
        perim     = cv2.arcLength(cnt, True)
        circ      = (4 * np.pi * area) / (perim ** 2) if perim > 0 else 0
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
# RUNTIME STATE
# ==========================================
robot_active   = False
left_flag      = 0
right_flag     = 0

# 3-state machine: "FOLLOW" | "HOLD"
robot_state    = "FOLLOW"
hold_until     = 0.0
cooldown_until = 0.0
locked_label   = "None"

label_history  = deque(maxlen=LABEL_HISTORY_LEN)

def get_stable_label():
    """Return (label, vote_count) for the most common non-None entry in history."""
    valid = [l for l in label_history if l != "None"]
    if not valid:
        return "None", 0
    label, count = Counter(valid).most_common(1)[0]
    return label, count

# ==========================================
# SYMBOL DETECTION FUNCTION  (upper ROI)
# ==========================================
def detect_symbol_in_roi(upper_gray, upper_sat, draw_frame, roi_y_offset):
    """
    Runs geometry + ORB pipelines on the upper ROI slice.
    Draws bounding boxes on draw_frame using full-frame coordinates.
    Returns the best recognised label string, or None.
    """
    blur_sym = cv2.GaussianBlur(upper_sat, (15, 15), 0)
    ret_sym, thresh_sym = cv2.threshold(blur_sym, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((30, 30), np.uint8)
    thresh_sym = cv2.morphologyEx(thresh_sym, cv2.MORPH_CLOSE, kernel)

    if ret_sym < 40:
        return None

    raw_contours, _ = cv2.findContours(thresh_sym, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sym_contours = merge_nearby_contours(raw_contours, proximity_threshold=60)

    best_label = None

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
            sol_diff    = abs(template["solidity"]     - live_solidity)
            ar_diff     = abs(template["aspect_ratio"] - live_ar) * 0.5
            circ_diff   = abs(template["circularity"]  - live_circ)
            extent_diff = abs(template["extent"]       - live_extent)
            score = hu_score + (sol_diff * 2.0) + ar_diff + circ_diff + (extent_diff * 3.0)
            if score < geo_best_score:
                geo_best_score = score
                geo_best_name  = template["name"]

        # --- ORB match ---
        y1 = max(0, y - 5);  y2 = min(upper_gray.shape[0], y + h + 5)
        x1 = max(0, x - 5);  x2 = min(upper_gray.shape[1], x + w + 5)
        roi_crop = upper_gray[y1:y2, x1:x2]

        live_kp, live_des = orb.detectAndCompute(roi_crop, None)
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
                    dst_pts = np.float32([live_kp[m.trainIdx].pt   for m in good]).reshape(-1, 1, 2)
                    M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M_hom is not None:
                        inliers = int(mask.ravel().sum())
                        if inliers > orb_best_inliers and inliers >= MIN_MATCH_COUNT:
                            orb_best_inliers = inliers
                            orb_best_name    = sym["name"]

        # --- Pick winner, draw in full-frame coords ---
        fy = y + roi_y_offset
        if orb_best_name != "Unknown Symbol":
            label      = f"{orb_best_name} ({orb_best_inliers} inliers)"
            box_color  = (255, 255, 0)
            best_label = orb_best_name
        elif geo_best_name != "Unknown":
            if "Arrow" in geo_best_name:
                geo_best_name = f"Arrow {get_arrow_direction(cnt)}"
                box_color = (0, 0, 255)
            else:
                box_color = (0, 255, 0)
            label      = f"{geo_best_name} ({geo_best_score:.2f})"
            best_label = geo_best_name
        else:
            continue   # unrecognised — skip

        cv2.rectangle(draw_frame, (x, fy), (x + w, fy + h), box_color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        text_x = x + (w - tw) // 2
        text_y = fy - 10 if fy - 10 > 10 else fy + h + 20
        cv2.putText(draw_frame, label, (text_x, text_y), font, 0.6, box_color, 2)

    return best_label


# ==========================================
# LINE FOLLOW FUNCTION  (lower ROI)
# ==========================================
def follow_line(lower_gray, draw_frame):
    """
    Runs PID line-follow on the lower ROI.
    Draws contour and centroid on draw_frame in full-frame coords.
    Returns (left_pwm, right_pwm).
    """
    global left_flag, right_flag

    blur = cv2.GaussianBlur(lower_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cx = LOWER_CENTRE
    cy = (LOWER_Y2 - LOWER_Y1) // 2

    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            left_flag  = 1 if cx <= LOWER_CENTRE else 0
            right_flag = 0 if cx <= LOWER_CENTRE else 1

        # Shift contour to full-frame coords for drawing
        shifted = largest + np.array([[[LOWER_X1, LOWER_Y1]]])
        cv2.drawContours(draw_frame, [shifted], -1, (0, 255, 0), 2)
    else:
        if left_flag == 1:
            cx = 0
        elif right_flag == 1:
            cx = LOWER_X2 - LOWER_X1

    cv2.circle(draw_frame, (cx + LOWER_X1, cy + LOWER_Y1), 5, (255, 0, 0), -1)
    return calculatePID(cx, dutyCycle)


# ==========================================
# MAIN LOOP
# ==========================================
print("\nReady. Press 's' to start, 'q' to quit.")

try:
    while True:
        now   = time.time()
        frame = picam2.capture_array()

        # Shared conversions
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, frame_sat, _ = cv2.split(frame_hsv)

        # Slice ROIs (no copy needed — read-only for detection)
        upper_gray = frame_gray[UPPER_Y1:UPPER_Y2, :]
        upper_sat  = frame_sat [UPPER_Y1:UPPER_Y2, :]
        lower_gray = frame_gray[LOWER_Y1:LOWER_Y2, LOWER_X1:LOWER_X2]

        # ==========================================
        # STATE: HOLD — stopped, counting down
        # ==========================================
        if robot_state == "HOLD":
            move(0, 0)
            remaining = max(0.0, hold_until - now)
            cv2.putText(frame, f"HOLD: {locked_label} ({remaining:.1f}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if now >= hold_until:
                robot_state    = "FOLLOW"
                cooldown_until = now + COOLDOWN_AFTER_HOLD
                label_history.clear()
                last_error = 0
                print(f"[RESUME] Cooldown {COOLDOWN_AFTER_HOLD}s before re-arming detection.")

        # ==========================================
        # STATE: FOLLOW — line follow + symbol scan
        # ==========================================
        else:
            left_pwm, right_pwm = follow_line(lower_gray, frame)

            # Symbol detection is gated by cooldown
            detected_label = None
            if now >= cooldown_until:
                detected_label = detect_symbol_in_roi(upper_gray, upper_sat, frame, UPPER_Y1)

            label_history.append(detected_label if detected_label else "None")
            stable_label, vote_count = get_stable_label()

            # Enough votes → transition to HOLD
            if stable_label != "None" and vote_count >= MIN_CONFIRM_COUNT:
                robot_state  = "HOLD"
                locked_label = stable_label
                hold_until   = now + STOP_DURATION
                label_history.clear()
                print(f"[SYMBOL CONFIRMED] {locked_label} "
                      f"({vote_count}/{LABEL_HISTORY_LEN} votes) — stopping {STOP_DURATION}s.")
            else:
                if robot_active:
                    move(left_pwm, right_pwm)
                else:
                    move(0, 0)
                    cv2.putText(frame, "STANDBY - Press 's' to start",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # HUD bottom bar
            in_cooldown = now < cooldown_until
            status = "COOLDOWN" if in_cooldown else f"{stable_label} {vote_count}/{LABEL_HISTORY_LEN}"
            cv2.putText(frame, f"FOLLOW | {status}",
                        (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # Draw ROI dividers
        cv2.line(frame, (0, UPPER_Y2), (frame_width, UPPER_Y2), (255, 255, 0), 1)
        cv2.line(frame, (0, LOWER_Y1), (frame_width, LOWER_Y1), (255, 255, 0), 1)
        cv2.rectangle(frame, (LOWER_X1, LOWER_Y1), (LOWER_X2, LOWER_Y2), (255, 0, 255), 1)

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