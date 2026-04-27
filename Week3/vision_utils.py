import cv2 as cv
import numpy as np
import multiprocessing as mp

def best_contour(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    c = max(contours, key=cv.contourArea)
    return c, cv.contourArea(c)

def _write_str(arr: mp.Array, text: str, max_bytes: int, lock: mp.Lock):
    with lock:
        enc = text.encode()[:max_bytes - 1]
        arr.raw = enc + b'\x00' * (max_bytes - len(enc))

def _read_str(arr: mp.Array, lock: mp.Lock) -> str:
    with lock:
        return arr.raw.rstrip(b'\x00').decode(errors='replace')

# Image recognition helper
def orb_match_symbol(bf, ref_entries, des_scene, threshold):
    for ref in ref_entries:
        if ref["des"] is None: continue
        matches = bf.knnMatch(ref["des"], des_scene, k=2)
        good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance)
        if good >= threshold: return True, good
    return False, 0

def _detect_shape(contour):
    area = cv.contourArea(contour)
    if area < 1500: return "Unknown", None

    # 🌟 NEW: Aspect Ratio Filter
    x, y, w, h = cv.boundingRect(contour)
    aspect_ratio = float(w) / h
    
    # A line will have an extreme aspect ratio (e.g., > 3.0 or < 0.33)
    # An arrow usually sits neatly between 0.5 and 2.0
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        return "Unknown", None  # It's too long/skinny, definitely a line!

    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.03 * peri, True)
    v = len(approx)
    
    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    if hull_area == 0: return "Unknown", None

    solidity = area / hull_area
    is_convex = cv.isContourConvex(approx)
    circ = 4 * np.pi * area / (peri * peri)
    
    # The rest of your Goldilocks math remains the same
    if 6 <= v <= 9 and not is_convex and 0.45 <= solidity <= 0.75 and circ >= 0.12:
        M = cv.moments(contour)
        if M["m00"] == 0: return "Arrow", "Unknown"
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])
        
        far = max(contour, key=lambda p: (p[0][0]-cx_)**2 + (p[0][1]-cy_)**2)
        dx, dy = far[0][0] - cx_, far[0][1] - cy_
        
        if abs(dx) > abs(dy): direction = "Left" if dx > 0 else "Right"
        else: direction = "Up" if dy > 0 else "Down"
        return "Arrow", direction
        
    return "Unknown", None