import cv2
import numpy as np
import config

class VisionAnalyser:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.MIN_MATCH_COUNT = 6
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
        for sym in config.SYMBOL_FILES:
            img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append({"name": sym["name"], "kp": kp, "des": des})
        return templates
    
    # process line on live feed (grayscale only)
    def process_line(self, bottom_roi, left_flag, right_flag):
        bottom_gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
        bottom_blur = cv2.GaussianBlur(bottom_gray, (5, 5), 0)
        _, thresh = cv2.threshold(bottom_blur, 40, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                if cx <= config.FRAME_CENTRE: 
                    left_flag, right_flag = 1, 0
                elif cx > config.FRAME_CENTRE: 
                    left_flag, right_flag = 0, 1
        else:
            # Memory fallback
            if left_flag == 1: cx = 0
            elif right_flag == 1: cx = config.FRAME_WIDTH
            else: cx = config.FRAME_CENTRE
        return cx, left_flag, right_flag
    
# vision_utils.py
import cv2
import numpy as np
import config

class VisionAnalyzer:
    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.MIN_MATCH_COUNT = 6
        self.arrow_templates = self._load_arrow_templates()
        self.orb_templates = self._load_orb_templates()

    def _load_arrow_templates(self):
        templates = []
        if config.ARROW_IMG_PATH:
            arrow_img = cv2.imread(config.ARROW_IMG_PATH)
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

    def _load_orb_templates(self):
        templates = []
        for sym in config.SYMBOL_FILES:
            img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append({"name": sym["name"], "kp": kp, "des": des})
        return templates

    def process_line(self, bottom_roi, left_flag, right_flag):
        """Processes the line ROI and returns the centroid (cx) and updated memory flags."""
        gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                if cx <= config.FRAME_CENTRE: 
                    left_flag, right_flag = 1, 0
                elif cx > config.FRAME_CENTRE: 
                    left_flag, right_flag = 0, 1
        else:
            # Memory fallback
            if left_flag == 1: cx = 0
            elif right_flag == 1: cx = config.FRAME_WIDTH
            else: cx = config.FRAME_CENTRE
            
        return cx, left_flag, right_flag

    def detect_symbol(self, top_roi):
        """Processes the symbol ROI and returns the detected symbol's name or None."""
        top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        _, top_sat, _ = cv2.split(cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV))
        ret_sym, top_thresh = cv2.threshold(cv2.GaussianBlur(top_sat, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if ret_sym < 40:
            return None

        top_thresh = cv2.morphologyEx(top_thresh, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
        sym_contours, _ = cv2.findContours(top_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sym_contours = self._merge_nearby_contours(sym_contours, proximity_threshold=60)

        for cnt in sym_contours:
            area = cv2.contourArea(cnt)
            if area < 500: continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # --- FEATURE 1: 5-Pixel Padding for ORB Context ---
            y1 = max(0, y - 5)
            y2 = min(top_gray.shape[0], y + h + 5)
            x1 = max(0, x - 5)
            x2 = min(top_gray.shape[1], x + w + 5)
            
            # Extract padded ROI
            roi_gray = top_gray[y1:y2, x1:x2]
            
            # A. Check ORB Templates First
            live_kp, live_des = self.orb.detectAndCompute(roi_gray, None)
            
            if live_des is not None and len(live_kp) > self.MIN_MATCH_COUNT:
                best_inliers, best_name = 0, ""
                
                for sym in self.orb_templates:
                    # --- FEATURE 2: Safety guard to prevent crash on bad templates ---
                    if sym["des"] is None or len(sym["des"]) < 2:
                        continue

                    matches = self.bf.knnMatch(sym["des"], live_des, k=2)
                    
                    good = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good.append(m)
                                
                    if len(good) >= self.MIN_MATCH_COUNT:
                        src_pts = np.float32([sym["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([live_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if M_hom is not None:
                            inliers = mask.sum()
                            if inliers > best_inliers and inliers >= self.MIN_MATCH_COUNT:
                                best_inliers, best_name = inliers, sym["name"]
                                
                if best_inliers > 0:
                    return best_name

            # B. Check Arrow Templates if ORB fails
            if self.arrow_templates:
                live_hull = cv2.convexHull(cnt)
                live_rect = cv2.minAreaRect(cnt)
                rect_w, rect_h = live_rect[1]
                live_sol = area / cv2.contourArea(live_hull) if cv2.contourArea(live_hull) > 0 else 0
                live_ar = max(rect_w, rect_h) / min(rect_w, rect_h) if min(rect_w, rect_h) > 0 else 0
                live_ext = area / (rect_w * rect_h) if (rect_w * rect_h) > 0 else 0
                live_circ = (4 * np.pi * area) / (cv2.arcLength(cnt, True) ** 2) if cv2.arcLength(cnt, True) > 0 else 0

                best_score = 1.5
                for temp in self.arrow_templates:
                    score = cv2.matchShapes(temp["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                    score += abs(temp["solidity"] - live_sol) * 2.0
                    score += abs(temp["aspect_ratio"] - live_ar) * 0.5
                    score += abs(temp["circularity"] - live_circ)
                    score += abs(temp["extent"] - live_ext) * 3.0
                    if score < best_score: best_score = score

                if best_score < 1.0:
                    direction = self._get_arrow_direction(cnt)
                    return f"ARROW_{direction}"

        return None

    @staticmethod
    def _get_arrow_direction(arrow_contour):
        x, y, w, h = cv2.boundingRect(arrow_contour)
        box_center_x, box_center_y = x + (w / 2.0), y + (h / 2.0)
        M = cv2.moments(arrow_contour)
        if M["m00"] == 0: return "UNKNOWN"
        dx = (M["m10"] / M["m00"]) - box_center_x
        dy = (M["m01"] / M["m00"]) - box_center_y
        if abs(dy) > abs(dx): return "DOWN" if dy > 0 else "UP"
        else: return "LEFT" if dx > 0 else "RIGHT"

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