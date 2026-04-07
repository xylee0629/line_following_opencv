import cv2
import numpy as np
import config

class VisionAnalyzer:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.MIN_MATCH_COUNT = 10
        self.arrow_templates = self._load_arrow_templates()
        self.orb_templates = self._load_orb_templates()
        
    # load and process arrow template
    def _load_arrow_templates(self):
        templates = []
        if config.ARROW_PATH:
            arrow_img = cv2.imread(config.ARROW_PATH)
            if arrow_img is not None:
                _, sat, _ = cv2.split(cv2.cvtColor(arrow_img, cv2.COLOR_BGR2HSV))
                _, thresh = cv2.threshold(cv2.GaussianBlur(sat, (15, 15), 0), 50, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        hull = cv2.convexHull(cnt)
                        rect = cv2.minAreaRect(cnt)
                        w, h = rect[1]
                        templates.append({
                            "contour": cnt,
                            "solidity": cv2.contourArea(cnt) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0,
                            "aspect_ratio": max(w, h) / min(w, h) if min(w, h) > 0 else 0,
                            "extent": cv2.contourArea(cnt) / (w * h) if (w * h) > 0 else 0,
                            "circularity": (4 * np.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0
                        })
        return templates

    # load and process orb template 
    def _load_orb_templates(self):
        templates = []
        for sym in config.SYMBOL_PATHS:
            img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append({"name": sym["name"], "kp": kp, "des": des})
        return templates
    
    
    def get_line_paths(self, bottom_roi):
        """Step 1: Extracts valid contours based on color priority and checks for splits."""
        bottom_hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)

        # 1. Define strict HSV color ranges
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        lower_red1 = np.array([0, 150, 130])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 150, 130])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([24, 150, 130])
        upper_yellow = np.array([36, 255, 255])

        # 2. Create individual masks & apply digital bezel
        mask_black = cv2.inRange(bottom_hsv, lower_black, upper_black)
        mask_red = cv2.bitwise_or(cv2.inRange(bottom_hsv, lower_red1, upper_red1), 
                                  cv2.inRange(bottom_hsv, lower_red2, upper_red2))
        mask_yellow = cv2.inRange(bottom_hsv, lower_yellow, upper_yellow)

        h, w = mask_black.shape
        cv2.rectangle(mask_black, (0, 0), (w, h), 0, thickness=35)
        cv2.rectangle(mask_red, (0, 0), (w, h), 0, thickness=35)
        cv2.rectangle(mask_yellow, (0, 0), (w, h), 0, thickness=35)

        # 3. Find contours
        contours_red, _ = cv2.findContours(cv2.GaussianBlur(mask_red, (5, 5), 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(cv2.GaussianBlur(mask_yellow, (5, 5), 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_black, _ = cv2.findContours(cv2.GaussianBlur(mask_black, (5, 5), 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Priority Logic
        target_contours = None
        detected_color = None
        MIN_AREA = 500 

        if len(contours_red) > 0 and max([cv2.contourArea(c) for c in contours_red]) > MIN_AREA:
            target_contours = contours_red
            detected_color = (0, 0, 255) 
        elif len(contours_yellow) > 0 and max([cv2.contourArea(c) for c in contours_yellow]) > MIN_AREA:
            target_contours = contours_yellow
            detected_color = (0, 255, 255) 
        elif len(contours_black) > 0 and max([cv2.contourArea(c) for c in contours_black]) > MIN_AREA:
            target_contours = contours_black
            detected_color = (0, 255, 0) 

# 5. Extract Valid Paths & Check for Split or Junction Shape
        valid_paths = []
        valid_junction = False
        
        if target_contours is not None:
            valid_paths = [c for c in target_contours if cv2.contourArea(c) > MIN_AREA]
            
            # --- JUNCTION DETECTION LOGIC ---
            
            # Condition 1: The Clean Split (The camera sees two distinct paths)
            if len(valid_paths) >= 2:
                valid_junction = True
                
            # Condition 2: The "Blob" Junction (The camera sees one shape, but it's a fork or crossroad)
            elif len(valid_paths) == 1:
                main_contour = valid_paths[0]
                
                # Get Width
                _, _, w, _ = cv2.boundingRect(main_contour)
                
                # Get Solidity
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                actual_area = cv2.contourArea(main_contour)
                solidity = actual_area / hull_area if hull_area > 0 else 0
                
                # If the line spans 75% of the screen OR its solidity drops below 0.70
                if w > config.FRAME_WIDTH * 0.75 or solidity < 0.70:
                    valid_junction = True

        return valid_paths, detected_color, valid_junction


    def calculate_line_center(self, valid_paths, detected_color, left_flag, right_flag, arrow_direction=None):
        """Step 2: Picks the correct path and calculates the steering centroid."""
        # Memory Fallback
        if left_flag == 1: cx = 0
        elif right_flag == 1: cx = config.FRAME_WIDTH
        else: cx = config.FRAME_CENTRE

        draw_data = None 

        if len(valid_paths) > 0:
            
            # --- THE ARROW SELECTION LOGIC ---
            if len(valid_paths) >= 2 and arrow_direction is not None:
                if arrow_direction == "LEFT":
                    # Sort contours by X-coordinate and pick the one furthest to the left
                    selected_contour = min(valid_paths, key=lambda c: cv2.boundingRect(c)[0])
                elif arrow_direction == "RIGHT":
                    # Sort contours by X-coordinate and pick the one furthest to the right
                    selected_contour = max(valid_paths, key=lambda c: cv2.boundingRect(c)[0])
                else:
                    # Fallback if arrow is unknown: pick largest
                    selected_contour = max(valid_paths, key=cv2.contourArea)
            else:
                # Default behavior: pick largest contour
                selected_contour = max(valid_paths, key=cv2.contourArea)

            # --- CENTROID MATH ---
            M = cv2.moments(selected_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) 

                if cx <= config.FRAME_CENTRE:  
                    left_flag, right_flag = 1, 0
                elif cx > config.FRAME_CENTRE:  
                    left_flag, right_flag = 0, 1
                
                draw_data = (selected_contour, detected_color, cx, cy)
                    
        return cx, left_flag, right_flag, draw_data
    
    
    
    def detect_symbol(self, top_roi):
        """Processes the symbol ROI and returns the symbol_name or None."""
        top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        _, top_sat, _ = cv2.split(cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV))
        ret_sym, top_thresh = cv2.threshold(cv2.GaussianBlur(top_sat, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if ret_sym < 40:
            return None, None

        top_thresh = cv2.morphologyEx(top_thresh, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
        sym_contours, _ = cv2.findContours(top_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sym_contours = self._merge_nearby_contours(sym_contours, proximity_threshold=60)

        for cnt in sym_contours:
            area = cv2.contourArea(cnt)
            if area < 500: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            box = (x, y, w, h)
            
            # A. Check Arrow Templates
            if self.arrow_templates:
                live_hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(live_hull)
                rect_w, rect_h = cv2.minAreaRect(cnt)[1]
                perimeter = cv2.arcLength(cnt, True)
                
                rect_area = rect_w * rect_h
                min_dim = min(rect_w, rect_h)
                
                live_sol = area / hull_area if hull_area > 0 else 0
                live_ar = max(rect_w, rect_h) / min_dim if min_dim > 0 else 0
                live_ext = area / rect_area if rect_area > 0 else 0
                live_circ = (12.566 * area) / (perimeter * perimeter) if perimeter > 0 else 0

                best_score = 1.5
                for temp in self.arrow_templates:
                    score = cv2.matchShapes(temp["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                    score += abs(temp["solidity"] - live_sol) * 2.0
                    score += abs(temp["aspect_ratio"] - live_ar) * 0.5
                    score += abs(temp["circularity"] - live_circ)
                    score += abs(temp["extent"] - live_ext) * 3.0
                    if score < best_score: 
                        best_score = score

                if best_score < 1.0:
                    direction = self._get_arrow_direction(cnt)
                    return f"ARROW_{direction}", box
                
            # B. Check ORB Templates
            y1 = max(0, y - 5)
            y2 = min(top_gray.shape[0], y + h + 5)
            x1 = max(0, x - 5)
            x2 = min(top_gray.shape[1], x + w + 5)
            
            roi_gray = top_gray[y1:y2, x1:x2]
            live_kp, live_des = self.orb.detectAndCompute(roi_gray, None)
            
            if live_des is not None and len(live_kp) > self.MIN_MATCH_COUNT:
                best_inliers, best_name = 0, ""
                
                for sym in self.orb_templates:
                    if sym["des"] is None or len(sym["des"]) < 2:
                        continue

                    matches = self.bf.knnMatch(sym["des"], live_des, k=2)
                    good = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.70 * n.distance:
                                good.append(m)
                                
                    if len(good) >= self.MIN_MATCH_COUNT:
                        src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M_hom is not None:
                            inliers = mask.sum()
                            inlier_ratio = inliers / len(good) if len(good) > 0 else 0
                            
                            if inliers > best_inliers and inliers >= self.MIN_MATCH_COUNT and inlier_ratio > 0.40:
                                best_inliers, best_name = inliers, sym["name"]
                                
                if best_inliers > 0:
                    return best_name, box
                    
        return None, None
    
    @staticmethod
    def _get_arrow_direction(arrow_contour):
        x, y, w, h = cv2.boundingRect(arrow_contour)
        M = cv2.moments(arrow_contour)
        if M["m00"] == 0: return "UNKNOWN"
        
        # Calculate horizontal displacement between center of mass and bounding box center
        dx = (M["m10"] / M["m00"]) - (x + w / 2.0)
        
        # Compare to determine orientation
        return "LEFT" if dx > 0 else "RIGHT"

    @staticmethod
    def _merge_nearby_contours(contours, proximity_threshold=60):
        if not contours: return []
        rects = [cv2.boundingRect(c) for c in contours]
        parent = list(range(len(contours)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        def union(i, j): parent[find(i)] = find(j)
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