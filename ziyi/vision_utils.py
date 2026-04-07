import cv2
import numpy as np
import config
from scipy.signal import convolve2d

class VisionAnalyzer:
    def __init__(self):
        self.orb = cv2.ORB_create()  # ORB detector
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # Matcher for ORB features
        self.MIN_MATCH_COUNT = 15  # Minimum number of matches for symbol detection
        self.arrow_templates = self._load_arrow_templates()
        self.orb_templates = self._load_orb_templates()
        
    # Wiener filter for deblurring motion blur
    def wiener_filter(self, image, noise_var=0.1, estimate_var=0.1):
        # Apply Non-Local Means Denoising for motion blur reduction
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 7, 21)

    # Load and process arrow templates for symbol detection
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

    # Load and process ORB templates for symbol matching
    def _load_orb_templates(self):
        templates = []
        for sym in config.SYMBOL_PATHS:
            img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append({"name": sym["name"], "kp": kp, "des": des})
        return templates

    # Detect symbols based on ORB matching
    def detect_symbols(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)

        if des is None:
            return []

        symbol_matches = []
        for template in self.orb_templates:
            # Match descriptors using BFMatcher
            matches = self.bf.match(des, template['des'])

            # Apply ratio test (Lowe's ratio test) to filter good matches
            good_matches = [m for m in matches if m.distance < 0.75 * min([m.distance for m in matches])]

            if len(good_matches) > self.MIN_MATCH_COUNT:
                # We have a match; record the symbol name and matched points
                symbol_matches.append({
                    "symbol": template["name"],
                    "matches": good_matches
                })

        return symbol_matches

    # Line following logic (process the image and detect lines)
    def process_line(self, bottom_roi, left_flag, right_flag):
        # Apply Wiener filter for deblurring
        filtered_image = self.wiener_filter(bottom_roi)

        # Convert to HSV
        bottom_hsv = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

        # 1. Define HSV color ranges (black, red, yellow)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])

        lower_red2 = np.array([160, 170, 100])
        upper_red2 = np.array([180, 255, 255])

        lower_yellow = np.array([22, 120, 100]) 
        upper_yellow = np.array([38, 255, 255])

        # 2. Create individual masks for each color
        mask_black = cv2.inRange(bottom_hsv, lower_black, upper_black)
        mask_red = cv2.inRange(bottom_hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(bottom_hsv, lower_yellow, upper_yellow)

        # 3. Apply morphological operations (optional, for noise reduction)
        kernel = np.ones((5,5), np.uint8)
        mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel)
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        # 4. Find contours for each mask
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ==========================================
        # 5. Priority Logic (Red -> Yellow -> Black)
        # ==========================================
        target_contours = None
        detected_color = None
        MIN_AREA = 500  # Set a minimum contour area

        if len(contours_red) > 0 and max([cv2.contourArea(c) for c in contours_red]) > MIN_AREA:
            target_contours = contours_red
            detected_color = (0, 0, 255)  # Red
        elif len(contours_yellow) > 0 and max([cv2.contourArea(c) for c in contours_yellow]) > MIN_AREA:
            target_contours = contours_yellow
            detected_color = (0, 255, 255)  # Yellow
        elif len(contours_black) > 0 and max([cv2.contourArea(c) for c in contours_black]) > MIN_AREA:
            target_contours = contours_black
            detected_color = (0, 255, 0)  # Green

        # ==========================================
        # 6. Process the detected contours and calculate the center of mass
        # ==========================================
        if target_contours is not None:
            largest_contour = max(target_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if cx <= config.FRAME_CENTRE:  
                    left_flag, right_flag = 1, 0
                elif cx > config.FRAME_CENTRE:  
                    left_flag, right_flag = 0, 1

                draw_data = (largest_contour, detected_color, cx, cy)
                return cx, left_flag, right_flag, draw_data
        return config.FRAME_CENTRE, left_flag, right_flag, None