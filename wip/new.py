import cv2
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. HELPERS
# ==========================================
def get_arrow_direction(arrow_contour):
    """Calculates direction by comparing Center of Mass to the Bounding Box center."""
    x, y, w, h = cv2.boundingRect(arrow_contour)
    box_center_x, box_center_y = x + (w / 2.0), y + (h / 2.0)

    M = cv2.moments(arrow_contour)
    if M["m00"] == 0: return "UNKNOWN"

    dx = (M["m10"] / M["m00"]) - box_center_x
    dy = (M["m01"] / M["m00"]) - box_center_y

    if abs(dy) > abs(dx): return "DOWN" if dy > 0 else "UP"
    else: return "LEFT" if dx > 0 else "RIGHT"

def merge_nearby_contours(contours, proximity_threshold=60):
    """Groups nearby contours to prevent fragmented bounding boxes (e.g., for QR codes)."""
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

# ==========================================
# 2. LOAD ARROW TEMPLATES
# ==========================================
arrow_templates = []
print("\n--- LEARNING ARROW TEMPLATES ---")
arrow_img = cv2.imread('/home/raspberrypi/line_following_opencv/images/arrow.png')

if arrow_img is not None:
    _, sat, _ = cv2.split(cv2.cvtColor(arrow_img, cv2.COLOR_BGR2HSV))
    _, thresh = cv2.threshold(cv2.GaussianBlur(sat, (15, 15), 0), 50, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        
        arrow_templates.append({
            "contour": cnt,
            "solidity": cv2.contourArea(cnt) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0,
            "aspect_ratio": max(w, h) / min(w, h) if min(w, h) > 0 else 0,
            "extent": cv2.contourArea(cnt) / (w * h) if (w * h) > 0 else 0,
            "circularity": (4 * np.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
        })
    print(f"Successfully learned {len(arrow_templates)} arrow variations.")
else:
    print("ERROR: Could not load arrow image.")

# ==========================================
# 3. LOAD ORB SYMBOLS
# ==========================================
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
print("\n--- LEARNING SYMBOLS (ORB) ---")
for sym in symbol_files:
    img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
    if img is not None:
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            orb_templates.append({"name": sym["name"], "kp": kp, "des": des})
            print(f" -> Learned: {sym['name']}")

# ==========================================
# 4. PICAMERA LIVE FEED
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
print("\nStarting live feed. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, sat, _ = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    # Preprocessing
    blur = cv2.GaussianBlur(sat, (15, 15), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
    if ret < 40: thresh = np.zeros_like(thresh)

    raw_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours = merge_nearby_contours(raw_contours, proximity_threshold=60)

    for cnt in merged_contours:
        area = cv2.contourArea(cnt)
        if area < 500: continue

        x, y, w, h = cv2.boundingRect(cnt)

        # --- CHECK 1: IS IT A SYMBOL? (ORB) ---
        y1, y2 = max(0, y - 5), min(frame_gray.shape[0], y + h + 5)
        x1, x2 = max(0, x - 5), min(frame_gray.shape[1], x + w + 5)
        
        live_kp, live_des = orb.detectAndCompute(frame_gray[y1:y2, x1:x2], None)
        orb_matched = False

        if live_des is not None and len(live_kp) > MIN_MATCH_COUNT:
            best_inliers, best_name = 0, ""
            
            for sym in orb_templates:
                matches = bf.knnMatch(sym["des"], live_des, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance] if len(matches[0]) == 2 else []

                if len(good) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        inliers = mask.sum()
                        if inliers > best_inliers and inliers >= MIN_MATCH_COUNT:
                            best_inliers = inliers
                            best_name = sym["name"]

            if best_inliers > 0:
                orb_matched = True
                label = f"{best_name} ({best_inliers})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                continue # Skip arrow check if it's a known symbol

        # --- CHECK 2: IS IT AN ARROW? (Geometry) ---
        if not orb_matched and arrow_templates:
            live_hull = cv2.convexHull(cnt)
            live_rect = cv2.minAreaRect(cnt)
            (rect_w, rect_h) = live_rect[1]
            
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

                if score < best_score:
                    best_score = score

            # If score is below threshold, it's considered an arrow
            if best_score < 1.0: 
                direction = get_arrow_direction(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Arrow {direction}", (x, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Detection (Symbols & Arrows)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()