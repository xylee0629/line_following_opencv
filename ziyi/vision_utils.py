import cv2
import numpy as np
import config

class VisionAnalyzer:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=2000)  # Increase number of features
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.MIN_MATCH_COUNT = 15
        self.arrow_templates = self._load_arrow_templates()
        self.orb_templates = self._load_orb_templates()

    # -----------------------------
    # Load arrow templates
    # -----------------------------
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

    # -----------------------------
    # Load ORB templates
    # -----------------------------
    def _load_orb_templates(self):
        templates = []
        for sym in config.SYMBOL_PATHS:
            img = cv2.imread(sym["filepath"], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                kp, des = self.orb.detectAndCompute(img, None)
                if des is not None:
                    templates.append({"name": sym["name"], "kp": kp, "des": des})
        return templates

    # -----------------------------
    # Wiener filter for motion blur
    # -----------------------------
    def wiener_filter(self, image, kernel_size=15, angle=0, K=0.01):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1
        kernel /= kernel_size
        center = (kernel_size // 2, kernel_size // 2)
        rot = cv2.getRotationMatrix2D(center, angle, 1)
        kernel = cv2.warpAffine(kernel, rot, (kernel_size, kernel_size))
        img_fft = np.fft.fft2(gray)
        kernel_fft = np.fft.fft2(kernel, s=gray.shape)
        wiener = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + K)
        result_fft = wiener * img_fft
        result = np.fft.ifft2(result_fft)
        result = np.abs(result)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # -----------------------------
    # Frame averaging to reduce blur
    # -----------------------------
    def average_frames(self, frames, alpha=0.8):
        avg_frame = frames[0]
        for frame in frames[1:]:
            avg_frame = cv2.addWeighted(avg_frame, alpha, frame, 1 - alpha, 0)
        return avg_frame

    # -----------------------------
    # Get line paths from bottom ROI
    # -----------------------------
    def process_line(self, bottom_roi, left_flag, right_flag):
        bottom_hsv = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        lower_red2 = np.array([160, 170, 100])
        upper_red2 = np.array([180, 255, 255])
        lower_yellow = np.array([22, 120, 100])
        upper_yellow = np.array([38, 255, 255])

        # Create individual masks
        mask_black = cv2.inRange(bottom_hsv, lower_black, upper_black)
        mask_red = cv2.inRange(bottom_hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(bottom_hsv, lower_yellow, upper_yellow)

        # Find contours
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Priority Logic: Red -> Yellow -> Black
        target_contours = None
        detected_color = None
        MIN_AREA = 500

        if len(contours_red) > 0 and max([cv2.contourArea(c) for c in contours_red]) > MIN_AREA:
            target_contours = contours_red
            detected_color = (0, 0, 255)  # Red
        elif len(contours_yellow) > 0 and max([cv2.contourArea(c) for c in contours_yellow]) > MIN_AREA:
            target_contours = contours_yellow
            detected_color = (0, 255, 255)  # Yellow
        elif len(contours_black) > 0 and max([cv2.contourArea(c) for c in contours_black]) > MIN_AREA:
            target_contours = contours_black
            detected_color = (0, 255, 0)  # Green

        # Process the line center and draw data
        if left_flag == 1:
            cx = 0
        elif right_flag == 1:
            cx = config.FRAME_WIDTH
        else:
            cx = config.FRAME_CENTRE

        draw_data = None  # Default to nothing if no line is found

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

                # Package the visual data to send to the main thread
                draw_data = (largest_contour, detected_color, cx, cy)

        return cx, left_flag, right_flag, draw_data

    # -----------------------------
    # Detect symbols (Arrow + ORB)
    # -----------------------------
    def detect_symbol(self, top_roi):
        # Apply Wiener filter to reduce blur
        top_roi = self.wiener_filter(top_roi)

        # Convert to grayscale and apply Gaussian Blur for template matching
        top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        _, top_sat, _ = cv2.split(cv2.cvtColor(top_roi, cv2.COLOR_BGR2HSV))
        ret, top_thresh = cv2.threshold(cv2.GaussianBlur(top_sat, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if ret < 40:
            return None, None
        top_thresh = cv2.morphologyEx(top_thresh, cv2.MORPH_CLOSE, np.ones((30, 30), np.uint8))
        sym_contours, _ = cv2.findContours(top_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in sym_contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            box = (x, y, w, h)

            # Arrow template matching
            if self.arrow_templates:
                best_score = 1.5
                for temp in self.arrow_templates:
                    score = cv2.matchShapes(temp["contour"], cnt, cv2.CONTOURS_MATCH_I1, 0)
                    if score < best_score:
                        best_score = score

                if best_score < 1.0:
                    return "ARROW", box

            # ORB template matching
            if self.orb_templates:
                best_score = 2.0
                for sym in self.orb_templates:
                    kp, des = sym["kp"], sym["des"]
                    orb_matches = self.orb.match(des, top_gray)
                    good_matches = [m for m in orb_matches if m.distance < 50]
                    if len(good_matches) >= self.MIN_MATCH_COUNT:
                        best_score = 0.5  # Adjustable threshold for ORB matching
                        return sym["name"], box

        return None, None