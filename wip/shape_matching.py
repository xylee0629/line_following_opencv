import cv2
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. DYNAMIC ARROW DIRECTION FUNCTION
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
        # Vertical (Positive dy means pulled down, since OpenCV Y=0 is top)
        if dy > 0:
            return "DOWN"
        else:
            return "UP"
    else:
        # Horizontal (Positive dx means pulled right)
        if dx > 0:
            return "RIGHT"
        else:
            return "LEFT"

# ==========================================
# 2. SETUP GEOMETRY FOR SHAPES & ARROWS
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

templates = []
reference_displays = [] 
print("\n--- LEARNING REFERENCE SHAPES ---")

for ref_data in reference_files:
    file_path = ref_data["filepath"]
    shape_names = ref_data["names"]
    
    ref_img = cv2.imread(file_path)
    if ref_img is None:
        print(f"ERROR: Could not load shape image at {file_path}. Skipping.")
        continue
        
    print(f"Processing: {file_path}")
    ref_copy = ref_img.copy() 
    
    # Preprocessing using Saturation to ignore white background
    ref_hsv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2HSV)
    _, ref_saturation, _ = cv2.split(ref_hsv)
    
    # Apply a stronger blur to match the live feed's bridge-building
    ref_blur = cv2.GaussianBlur(ref_saturation, (15, 15), 0)
    ret, ref_threshold = cv2.threshold(ref_blur, 50, 255, cv2.THRESH_BINARY)
    
    # --- MATCHING THE LIVE KERNEL ---
    kernel = np.ones((25, 25), np.uint8)
    ref_threshold = cv2.morphologyEx(ref_threshold, cv2.MORPH_CLOSE, kernel)
    # --------------------------------
    
    ref_contours, _ = cv2.findContours(ref_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [c for c in ref_contours if 100 < cv2.contourArea(c) < (ref_img.shape[0]*ref_img.shape[1]*0.9)]
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1] // 50, cv2.boundingRect(c)[0]))

    for i, cnt in enumerate(valid_contours):
        if i < len(shape_names):
            name = shape_names[i]
        else:
            name = f"Extra Shape {i}"
            
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
        
        templates.append({
            "name": name, 
            "contour": cnt, 
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "extent": extent
        })
        print(f"  -> Learned: {name}")
        
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(ref_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(ref_copy, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    window_name = f"Reference: {file_path.split('/')[-1]}"
    reference_displays.append((window_name, ref_copy))

print(f"\nSuccessfully learned {len(templates)} master shape templates.")

# ==========================================
# 3. PICAMERA LIVE FEED
# ==========================================
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 360)}))
picam2.start()

print("\nStarting live feed. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    
    # Needs HSV for Geometry thresholding
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, frame_saturation, _ = cv2.split(frame_hsv)
    
    # Increase blur from (7,7) to (15,15) to help bridge gaps naturally
    frame_blur = cv2.GaussianBlur(frame_saturation, (15, 15), 0) 
    ret, frame_threshold = cv2.threshold(frame_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use kernel to close gaps
    kernel = np.ones((25, 25), np.uint8)
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
    
    # The "Otsu Trap" Sanity Check: Ignore empty frames
    if ret < 40:
        frame_threshold = np.zeros_like(frame_threshold)
        
    frame_contours, _ = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in frame_contours:
        # --- NOISE FILTER ---
        area = cv2.contourArea(cnt)
        if area < 500: # Ignore tiny background noise
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        best_match_name = "Unknown"
        best_match_score = 1.5 
        
        # ==========================================
        # STAGE 1: GEOMETRY MATCHING
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

        for template in templates:
            hu_score = cv2.matchShapes(template["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
            solidity_diff = abs(template["solidity"] - live_solidity)
            ar_diff = abs(template["aspect_ratio"] - live_aspect_ratio) * 0.5
            circ_diff = abs(template["circularity"] - live_circularity)
            extent_diff = abs(template["extent"] - live_extent)
            
            total_score = hu_score + (solidity_diff * 2.0) + ar_diff + circ_diff + (extent_diff * 3.0)
            
            if total_score < best_match_score:
                best_match_score = total_score
                best_match_name = template["name"]

        # ==========================================
        # STAGE 2: DRAWING & LABELING
        # ==========================================
        if best_match_name != "Unknown":
            display_name = best_match_name
            
            if "Arrow" in best_match_name:
                direction = get_arrow_direction(cnt)
                display_name = f"Arrow {direction}"
                box_color = (0, 0, 255) # Red for Arrows
                label = f"{display_name} ({best_match_score:.2f})"
            else:
                box_color = (0, 255, 0) # Green for Basic Shapes
                label = f"{display_name} ({best_match_score:.2f})"
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x + (w - text_width) // 2
            text_y = y + (h + text_height) // 2
            
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, box_color, thickness)
            
    # ==========================================
    # 4. DISPLAY FEED & WINDOWS
    # ==========================================
    for window_name, ref_img_display in reference_displays:
        cv2.imshow(window_name, ref_img_display)

    cv2.imshow("Threshold (Saturation)", frame_threshold) 
    cv2.imshow("Live Geometry Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
picam2.stop()       
cv2.destroyAllWindows()

