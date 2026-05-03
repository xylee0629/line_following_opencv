import cv2 
import numpy as np
import time
from collections import deque
from multiprocessing import shared_memory
import traceback

from config import *
from vision_utils import _write_str, orb_match_symbol, detect_arrow

def image_worker(shm_name, frame_lock, img_ready_event, string_lock, out_is_priority, out_found, out_instruction, out_instruction_ready, img_display_shm_name, img_display_lock):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_bf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
        display_shm = shared_memory.SharedMemory(name=img_display_shm_name)
        display_buf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=display_shm.buf)

        # Initialise orb matching 
        orb = cv2.ORB_create(nfeatures=500, nlevels=8, fastThreshold=17)
        bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        reference_data = []
        for symbol_id, (img_files, threshold) in SYMBOL_DICT.items():
            refs = []
            for img_file in img_files:
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    kp, des = orb.detectAndCompute(img, None)
                    refs.append({"filename": img_file, "kp": kp, "des": des})
            reference_data.append({"id": symbol_id, "name": SYMBOL_NAME[symbol_id], "threshold": threshold, "refs": refs})

        ref_by_id = {entry["id"]: entry for entry in reference_data}

        label_history = deque(maxlen = DEBOUNCE_FRAMES)
        cooldown_counter = 0
        missed_frames = 0

        while True:
            img_ready_event.wait()
            img_ready_event.clear()

            with frame_lock:
                frame = frame_bf.copy()

            display_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # frame in format of BRG

            # Local variables 
            current_priority = out_is_priority.value
            found = False
            label = ""
            instruction = ""
            best_contour_for_display = None 

            if not current_priority:
                # Find possible symbols through colour first
                blur = cv2.GaussianBlur(frame, (3,3), 0)
                
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB) # LAB is a colour space that has values for lightness, making it tolerable to light changes

                # Create dict for possible candidates
                all_candidates = []
                for colour_name, parameters in IMAGE_COLOUR_RANGES.items():
                    if parameters["space"] == "HSV":
                        src = hsv
                    else:
                        src = lab
                    mask = cv2.inRange(src, parameters["lower"], parameters["upper"])
                    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                        a = cv2.contourArea(cnt)
                        if a > 1000:
                            all_candidates.append((a, cnt, colour_name))
                            # if area of the found contour is larger than 1000 px, append tuple of the area, contour coordinate and colour
                all_candidates.sort(key=lambda x:x[0], reverse=True) # sort by the first element in the tuple (area of the contour), largest to smallest
                top_candidates = all_candidates[:3] # top candidates is the first three elements of all

                colour_to_ids = {"Yellow": [4], "Blue/Teal": [1, 2], "Purple": [1, 2], "Green": [0, 3]}
                
                # Draw bounding rectangle around the region of interest
                for area, cnt, colour in top_candidates:
                    if colour in colour_to_ids:
                        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
                        x, y, w, h = cv2.boundingRect(cnt)
                        # include padding so that the full symbol is in the ROI
                        pad = 15
                        y1, y2 = max(0, y-pad), min(FRAME_HEIGHT, y+h+pad)
                        x1, x2 = max(0, x-pad), min(FRAME_WIDTH, x+w+pad)

                        roi = grey[y1:y2, x1:x2]
                        _, des_s = orb.detectAndCompute(roi, None)

                        if des_s is not None:
                            for sym_id in colour_to_ids[colour]:
                                entry = ref_by_id[sym_id]
                                matched, good_count = orb_match_symbol(bf, entry["refs"], des_s, entry["threshold"])
                                if matched:
                                    label = entry["name"]; 
                                    found = True; 
                                    best_contour_for_display = cnt
                                    break 
                    if not found:
                        shape, direction = detect_arrow(cnt)
                        if shape != "Unknown":
                            label = shape + (f" ({direction})" if direction else "")
                            found = True; best_contour_for_display = cnt
                    if found: break 

                # If the robot sees a symbol
            if found and cooldown_counter == 0:
                # Add the recorded symbol to history
                label_history.append(label)
                missed_frames = 0
                # If the robot sees the symbol for debounce frames number of times 
                if len(label_history) == label_history.maxlen and len(set(label_history)) == 1:
                    confirmed_label = label_history[0]
                    instruction = LABEL_TO_INSTRUCTION.get(confirmed_label, "")
                    label_history.clear()
                    cooldown_counter = 8
            else:
                # If the robot doesn't see the symbol/ sees another symbol
                if cooldown_counter > 0: cooldown_counter -= 1
                missed_frames += 1
                # If symbol isn't seen for more than 4 frames 
                if missed_frames > 4: 
                    # Clear history
                    label_history.clear()
                    instruction = ""

            if current_priority:
                cv2.putText(display_bgr, "Line Following Only", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if best_contour_for_display is not None:
                x, y, w, h = cv2.boundingRect(best_contour_for_display)
                cv2.rectangle(display_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
                if label: cv2.putText(display_bgr, label, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            elif label:
                cv2.putText(display_bgr, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            with img_display_lock:
                np.copyto(display_buf, display_bgr)

            out_found.value = found
            if instruction:
                _write_str(out_instruction, instruction, 32, string_lock)
                out_instruction_ready.value = True

    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
