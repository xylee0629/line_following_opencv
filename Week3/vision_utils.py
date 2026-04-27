import cv2
import numpy as np
import multiprocessing as mp

from config import *

def bestContour(mask):
    # Finds the largest contour detected in frame
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours:
        return None, 0
    # returns the coordinates of the contour, using the area as the sorting criteria
    c = max(contours, key=cv2.contourArea)
    return c, cv2.contourArea(c)

def _write_str(arr: mp.Array, text: str, max_bytes: int, lock: mp.Lock):
    with lock:
        enc = text.encode()[:max_bytes - 1]
        arr.raw = enc + b'\x00' * (max_bytes - len(enc))

def _read_str(arr: mp.Array, lock: mp.Lock) -> str:
    with lock:
        return arr.raw.rstrip(b'\x00').decode(errors='replace')
    
def orb_match_symbol(bf, ref_entries, des_scene, threshold):
    for ref in ref_entries:
        if ref["des"] is None: continue
        matches = bf.knnMatch(ref["des"], des_scene, k=2)
        good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance)
        if good >= threshold: return True, good
    return False, 0

def detect_arrow(contour):
    area = cv2.contourArea(contour)
    if area < 1500:
        return "Unknown", None
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    # if aspect ratio is too high (width too big) or too small (width too narrow/ large height), it's probably not an arrow
    if aspect_ratio > 2.5 or aspect_ratio < 0.4:
        return "Unknown", None
    
    perimeter = cv2.arcLength(contour, True)
    # Reduces the number of vertices 
    approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    V = len(approx) # How many vertices left

    # Convex Hull: minimum boundary that can completely enclose or wrap the object
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0: return "Unknown", None

    solidity = area / hull_area
    is_convex = cv2.isContourConvex(approx) # Checks if the contour is convex (has no interior angle > 180 degrees)
    circle = 4 * np.pi * area / (perimeter * perimeter)

    if 6 <= V <= 9 and not is_convex and 0.45 <= solidity <= 0.75 and circle >= 0.12:
        M = cv2.moments(contour)
        if M["m00"] == 0: 
            return "Arrow", "Unknown"
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])
        
        # Searches through all points in the contour to find the point furthest from centre of mass 
        far = max(contour, key=lambda p: (p[0][0]-cx_)**2 + (p[0][1]-cy_)**2)
        dx, dy = far[0][0] - cx_, far[0][1] - cy_ # Finds the difference between the furthest point and centre of mass of arrow
        
        if abs(dx) > abs(dy): direction = "Left" if dx > 0 else "Right"
        else: direction = "Up" if dy > 0 else "Down"
        return "Arrow", direction
        
    return "Unknown", None