import cv2 as cv
import numpy as np
import time
from collections import deque
from multiprocessing import shared_memory
import traceback

from week3.config import *
from week3.vision_utils import orb_match_symbol, _detect_shape, _write_str

def image_worker(
    shm_name, frame_lock, img_ready_event, string_lock, out_found, out_label, out_instruction, 
    out_instruction_ready, disp_shm_name, disp_lock, out_is_priority, out_investigating
):
    try:
        shm  = shared_memory.SharedMemory(name=shm_name)
        fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
        disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
        disp_buf = np.ndarray(IMG_DISPLAY_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

        orb = cv.ORB_create(nfeatures=500, nlevels=8, fastThreshold=17)
        bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

        reference_data = []
        for symbol_id, (img_files, threshold) in SYMBOL_DICT.items():
            refs = []
            for img_file in img_files:
                img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    refs.append({"filename": img_file, "kp": kp, "des": des})
            reference_data.append({"id": symbol_id, "name": SYMBOL_NAME[symbol_id], "threshold": threshold, "refs": refs})
     
        ref_by_id = {entry["id"]: entry for entry in reference_data}
        label_history = deque(maxlen = DEBOUNCE_FRAMES) 

        cooldown_counter, missed_frames = 0, 0

        while True:
            # Wait for a new frame efficiently
            img_ready_event.wait()
            img_ready_event.clear()

            with frame_lock: frame = fbuf.copy()
            display_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            found, is_investigating = False, False
            label, instruction = "", ""
            best_contour_for_display = None
            current_is_priority = bool(out_is_priority.value)
            
            # CPU SAVER
            # Only do the heavy color math and shape detection if we ARE NOT 
            # currently dealing with a red/yellow priority line.
            if not current_is_priority:
                gray_scene = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) 
                blurred = cv.GaussianBlur(frame, (3, 3), 0)
                HSV = cv.cvtColor(blurred, cv.COLOR_RGB2HSV)
                LAB = cv.cvtColor(blurred, cv.COLOR_RGB2LAB)
                
                all_candidates = []
                for colour_name, params in IMAGE_COLOUR_RANGES.items():
                    src  = HSV if params["space"] == "HSV" else LAB
                    mask = cv.inRange(src, params["lower"], params["upper"])
                    for cnt in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
                        a = cv.contourArea(cnt)
                        # Slightly lowered area threshold since robot is slower
                        if a >= 1000: all_candidates.append((a, cnt, colour_name))
                
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = all_candidates[:3]

                colour_to_ids = {"Yellow": [4], "Blue/Teal": [1, 2], "Purple": [1, 2], "Green": [0, 3]}

                for area, cnt, detected_colour in top_candidates: 
                    is_investigating = True 

                    if detected_colour in colour_to_ids:
                        x, y, w, h = cv.boundingRect(cnt)
                        pad = 15
                        y1, y2 = max(0, y-pad), min(FRAME_HEIGHT, y+h+pad)
                        x1, x2 = max(0, x-pad), min(FRAME_WIDTH, x+w+pad)
                        
                        roi_gray = gray_scene[y1:y2, x1:x2]
                        _, des_s = orb.detectAndCompute(roi_gray, None)

                        if des_s is not None:
                            for sym_id in colour_to_ids[detected_colour]:
                                entry = ref_by_id[sym_id]
                                matched, good_count = orb_match_symbol(bf, entry["refs"], des_s, entry["threshold"])
                                if matched:
                                    label = entry["name"]; found = True; best_contour_for_display = cnt
                                    break 
                    
                    if not found:
                        shape, direction = _detect_shape(cnt)
                        if shape != "Unknown":
                            label = shape + (f" ({direction})" if direction else "")
                            found = True; best_contour_for_display = cnt
                    if found: break 

            if found and cooldown_counter == 0:
                label_history.append(label)
                missed_frames = 0
                if len(label_history) == label_history.maxlen and len(set(label_history)) == 1:
                    confirmed_label = label_history[0]
                    instruction = LABEL_TO_INSTRUCTION.get(confirmed_label, "")
                    label_history.clear()
                    cooldown_counter = 8
            else:
                if cooldown_counter > 0: cooldown_counter -= 1
                missed_frames += 1
                if missed_frames > 4: label_history.clear(); instruction = ""   

            if current_is_priority:
                cv.putText(display_bgr, "Line Following Only", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
            if best_contour_for_display is not None:
                x, y, w, h = cv.boundingRect(best_contour_for_display)
                cv.rectangle(display_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
                if label: cv.putText(display_bgr, label, (x, max(10, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif label:
                cv.putText(display_bgr, label, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            with disp_lock: np.copyto(disp_buf, display_bgr)

            out_found.value = found
            out_investigating.value = is_investigating 
            
            # Use lock to write strings safely
            _write_str(out_label, label, 64, string_lock)
            if instruction:
                _write_str(out_instruction, instruction, 32, string_lock)
                out_instruction_ready.value = True

    except Exception as e:
        print(f"\n[img_worker] CRASHED: {e}\n", flush=True)
        traceback.print_exc()