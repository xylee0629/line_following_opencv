import cv2
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. ARROW DIRECTION HELPER
# ==========================================
def get_arrow_direction(arrow_contour):
    """Calculates direction by comparing Center of Mass to the Bounding Box center."""
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


# ==========================================
# 2. CONTOUR CLUSTERING HELPER
# ==========================================
def merge_nearby_contours(contours, frame_shape, proximity_threshold=80):
    """
    Groups contours whose bounding boxes are within proximity_threshold pixels
    of each other, then returns one merged contour per group (via convex hull).
    Leaves isolated contours unchanged so normal shapes are unaffected.
    """
    if len(contours) == 0:
        return contours

    # Represent each contour as its bounding rect
    rects = [cv2.boundingRect(c) for c in contours]

    # Union-Find to cluster overlapping/nearby rects
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
            # Expand each rect by proximity_threshold before checking overlap
            if (xi - proximity_threshold < xj + wj and
                    xi + wi + proximity_threshold > xj and
                    yi - proximity_threshold < yj + hj and
                    yi + hi + proximity_threshold > yj):
                union(i, j)

    # Group contours by cluster
    clusters = {}
    for i, cnt in enumerate(contours):
        root = find(i)
        clusters.setdefault(root, []).append(cnt)

    merged = []
    for group in clusters.values():
        if len(group) == 1:
            # Isolated contour — pass through unchanged
            merged.append(group[0])
        else:
            # Multiple nearby contours — combine all points and take convex hull
            all_pts = np.vstack(group)
            hull = cv2.convexHull(all_pts)
            merged.append(hull)

    return merged


# ==========================================
# 3. GEOMETRY TEMPLATES (Shapes & Arrows)
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
reference_displays = []
print("\n--- LEARNING REFERENCE SHAPES (GEOMETRY) ---")

for ref_data in reference_files:
    file_path = ref_data["filepath"]
    shape_names = ref_data["names"]

    ref_img = cv2.imread(file_path)
    if ref_img is None:
        print(f"ERROR: Could not load shape image at {file_path}. Skipping.")
        continue

    print(f"Processing: {file_path}")
    ref_copy = ref_img.copy()

    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    _, ref_saturation, _ = cv2.split(ref_hsv)
    ref_blur = cv2.GaussianBlur(ref_saturation, (15, 15), 0)
    _, ref_threshold = cv2.threshold(ref_blur, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((25, 25), np.uint8)
    ref_threshold = cv2.morphologyEx(ref_threshold, cv2.MORPH_CLOSE, kernel)

    ref_contours, _ = cv2.findContours(ref_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in ref_contours if 100 < cv2.contourArea(c) < (ref_img.shape[0] * ref_img.shape[1] * 0.9)]
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0]))

    for i, cnt in enumerate(valid_contours):
        name = shape_names[i] if i < len(shape_names) else f"Extra Shape {i}"

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = cv2.contourArea(cnt) / hull_area if hull_area > 0 else 0

        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        rect_area = w * h
        extent = cv2.contourArea(cnt) / rect_area if rect_area > 0 else 0

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * cv2.contourArea(cnt)) / (perimeter ** 2) if perimeter > 0 else 0

        geo_templates.append({
            "name": name,
            "contour": cnt,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "extent": extent
        })
        print(f"  -> Learned: {name}")

        bx, by, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(ref_copy, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
        cv2.putText(ref_copy, name, (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    window_name = f"Reference: {file_path.split('/')[-1]}"
    reference_displays.append((window_name, ref_copy))

print(f"\nSuccessfully learned {len(geo_templates)} geometry templates.")


# ==========================================
# 3. ORB TEMPLATES (Complex Symbols)
# ==========================================
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
MIN_MATCH_COUNT = 6

symbol_files = [
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/button.jpg',      "name": "Button"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/fingerprint.jpg', "name": "Fingerprint"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/hazard.jpg',      "name": "Hazard"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/qr.jpg',          "name": "QR Code"},
    {"filepath": '/home/raspberrypi/line_following_opencv/images/symbols/recycle.jpg',     "name": "Recycle"}
]

orb_templates = []
print("\n--- LEARNING REFERENCE SYMBOLS (ORB) ---")

for sym_data in symbol_files:
    file_path = sym_data["filepath"]
    name = sym_data["name"]

    sym_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if sym_img is None:
        print(f"ERROR: Could not load symbol image at {file_path}. Skipping.")
        continue

    print(f"Processing: {file_path}")
    kp, des = orb.detectAndCompute(sym_img, None)

    if des is not None:
        orb_templates.append({"name": name, "kp": kp, "des": des})
        print(f"  -> Learned: {name} ({len(kp)} keypoints)")
    else:
        print(f"  -> Failed to find keypoints in {name}")

print(f"\nSuccessfully learned {len(orb_templates)} ORB symbol templates.")


# ==========================================
# 4. PICAMERA LIVE FEED
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

print("\nStarting combined live feed. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()

    # --- Shared preprocessing ---
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, frame_saturation, _ = cv2.split(frame_hsv)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_blur = cv2.GaussianBlur(frame_saturation, (15, 15), 0)
    ret, frame_threshold = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((30, 30), np.uint8S)
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)

    if ret < 40:
        frame_threshold = np.zeros_like(frame_threshold)

    raw_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Merge contours that are close together (fixes QR code splitting into sub-boxes)
    # proximity_threshold=60 bridges QR finder-pattern gaps without touching spaced shapes
    frame_contours = merge_nearby_contours(raw_contours, frame.shape, proximity_threshold=60)

    for cnt in frame_contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ==========================================
        # PIPELINE A: GEOMETRY MATCHING
        # ==========================================
        live_hull = cv2.convexHull(cnt)
        live_hull_area = cv2.contourArea(live_hull)
        live_solidity = area / live_hull_area if live_hull_area > 0 else 0

        live_rect = cv2.minAreaRect(cnt)
        (rect_w, rect_h) = live_rect[1]
        live_aspect_ratio = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
        live_rect_area = rect_w * rect_h
        live_extent = area / live_rect_area if live_rect_area > 0 else 0

        live_perimeter = cv2.arcLength(cnt, True)
        live_circularity = (4 * np.pi * area) / (live_perimeter ** 2) if live_perimeter > 0 else 0

        geo_best_name = "Unknown"
        geo_best_score = 1.5

        for template in geo_templates:
            hu_score = cv2.matchShapes(template["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
            solidity_diff = abs(template["solidity"] - live_solidity)
            ar_diff = abs(template["aspect_ratio"] - live_aspect_ratio) * 0.5
            circ_diff = abs(template["circularity"] - live_circularity)
            extent_diff = abs(template["extent"] - live_extent)

            total_score = hu_score + (solidity_diff * 2.0) + ar_diff + circ_diff + (extent_diff * 3.0)

            if total_score < geo_best_score:
                geo_best_score = total_score
                geo_best_name = template["name"]

        # ==========================================
        # PIPELINE B: ORB SYMBOL MATCHING
        # ==========================================
        y1 = max(0, y - 5);          y2 = min(frame_gray.shape[0], y + h + 5)
        x1 = max(0, x - 5);          x2 = min(frame_gray.shape[1], x + w + 5)
        roi_gray = frame_gray[y1:y2, x1:x2]

        live_kp, live_des = orb.detectAndCompute(roi_gray, None)

        orb_best_name = "Unknown Symbol"
        orb_best_inliers = 0

        if live_des is not None and len(live_kp) > MIN_MATCH_COUNT:
            for sym in orb_templates:
                if sym["des"] is None or len(sym["des"]) < 2:
                    continue

                matches = bf.knnMatch(sym["des"], live_des, k=2)

                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)

                if len(good_matches) >= MIN_MATCH_COUNT:
                    src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M_hom is not None:
                        valid_inliers = sum(mask.ravel().tolist())
                        if valid_inliers > orb_best_inliers and valid_inliers >= MIN_MATCH_COUNT:
                            orb_best_inliers = valid_inliers
                            orb_best_name = sym["name"]

        # ==========================================
        # PIPELINE DECISION & LABELING
        # Choose ORB result if it found a confident symbol match,
        # otherwise fall back to geometry result.
        # ==========================================
        geo_matched = geo_best_name != "Unknown"
        orb_matched = orb_best_name != "Unknown Symbol"

        if orb_matched:
            # ORB wins — complex symbol detected
            label = f"{orb_best_name} ({orb_best_inliers} inliers)"
            box_color = (255, 255, 0)  # Cyan
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            text_x = x + (w - tw) // 2
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(frame, label, (text_x, text_y), font, 0.6, box_color, 2)
            cv2.drawKeypoints(frame[y1:y2, x1:x2], live_kp, frame[y1:y2, x1:x2], color=(0, 255, 0), flags=0)

        elif geo_matched:
            # Geometry pipeline — shape or arrow
            display_name = geo_best_name
            if "Arrow" in geo_best_name:
                direction = get_arrow_direction(cnt)
                display_name = f"Arrow {direction}"
                box_color = (0, 0, 255)  # Red
            else:
                box_color = (0, 255, 0)  # Green

            label = f"{display_name} ({geo_best_score:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
            text_x = x + (w - tw) // 2
            text_y = y + (h + th) // 2
            cv2.putText(frame, label, (text_x, text_y), font, 0.6, box_color, 2)

    # ==========================================
    # 5. DISPLAY
    # ==========================================
    for window_name, ref_img_display in reference_displays:
        cv2.imshow(window_name, ref_img_display)

    cv2.imshow("Threshold", frame_threshold)
    cv2.imshow("Live Detection (Geometry + ORB)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
