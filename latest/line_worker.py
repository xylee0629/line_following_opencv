import cv2 as cv
import numpy as np
import time
from multiprocessing import shared_memory
import traceback

from config import *
from vision_utils import best_contour

def line_worker(
    shm_name, frame_lock, line_ready_event, out_reset_pid, out_pid, out_cx, out_cy, 
    out_has_line, out_lineArea, out_is_priority, out_turn_cmd, disp_shm_name, disp_lock
):
    try:
        shm  = shared_memory.SharedMemory(name=shm_name)
        fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)
        disp_shm  = shared_memory.SharedMemory(name=disp_shm_name)
        disp_buf  = np.ndarray(LINE_DISPLAY_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

        pid_state = {'last_error': 0.0, 'integral': 0.0, 'last_time': time.monotonic()}
        pid_out, cx, cy = 0.0, X_CENTRE, 150          
        prev_fps_time, smoothed_fps = time.monotonic(), 0.0
        lane_memory, red_left_votes, red_right_votes = None, 0, 0
        was_on_red, was_on_yellow = False, False

        def get_bbox(cnt): return cv.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)

        while True:
            # Wait for a new frame efficiently
            line_ready_event.wait()
            line_ready_event.clear()

            # Handle post-sleep PID resets to avoid windup
            if out_reset_pid.value:
                pid_state['last_error'] = 0.0
                pid_state['integral'] = 0.0
                pid_state['last_time'] = time.monotonic()
                out_reset_pid.value = False

            now = time.monotonic()
            dt_fps = now - prev_fps_time
            prev_fps_time = now
            if dt_fps > 0: smoothed_fps = (0.9 * smoothed_fps) + (0.1 * (1.0 / dt_fps))

            with frame_lock:
                crop_rgb = fbuf[LINE_CROP_START:FRAME_HEIGHT, :].copy()

            crop_bgr = cv.cvtColor(crop_rgb, cv.COLOR_RGB2BGR)
            blur = cv.GaussianBlur(crop_rgb, (3, 3), 0)
            hsv  = cv.cvtColor(blur, cv.COLOR_RGB2HSV)
            
            cnts, follow_colour, draw_colour = [], "None", (0, 255, 0)

            mask_red = cv.bitwise_or(cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_1"], LINE_COLOUR_RANGES["Red"]["upper_1"]), 
                                     cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_2"], LINE_COLOUR_RANGES["Red"]["upper_2"]))
            cnt_red, area_red = best_contour(mask_red)
            xr, yr, wr, hr = get_bbox(cnt_red)

            if area_red > 4500 and hr > 20 and wr > 20:
                cnts, follow_colour, draw_colour = [cnt_red], "Red", (0, 0, 255)
            else:
                mask_yellow = cv.inRange(hsv, LINE_COLOUR_RANGES["Yellow"]["lower"], LINE_COLOUR_RANGES["Yellow"]["upper"])
                cnt_yellow, area_yellow = best_contour(mask_yellow)
                xy, yy, wy, hy = get_bbox(cnt_yellow)
                
                if area_yellow > 4500 and hy > 135 and wy > 20:
                    cnts, follow_colour, draw_colour = [cnt_yellow], "Yellow", (0, 255, 255)
                    if not was_on_yellow: was_on_yellow = True
                else:
                    mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])
                    cnt_black, area_black = best_contour(mask_black)
                    if area_black > 4500:
                        cnts, follow_colour, draw_colour = [cnt_black], "Black", (0, 255, 0)
                
            if follow_colour != "Yellow": was_on_yellow = False
            has_line, current_area = False, 0.0
            
            if cnts:
                has_line = True
                largest_contour = max(cnts, key=cv.contourArea)
                current_area = cv.contourArea(largest_contour)

                x, y, w, h = cv.boundingRect(largest_contour)
                M = cv.moments(largest_contour)
                if M['m00'] != 0:
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + (w // 2), y + (h // 2)

                cv.drawContours(crop_bgr, [largest_contour], -1, draw_colour, 2)

                error = X_CENTRE - cx
                now   = time.monotonic()
                dt    = now - pid_state['last_time']
                pid_state['last_time'] = now
                    
                P = KP * error
                pid_state['integral'] += error * dt
                
                # Clamp the integral to prevent PID windup
                pid_state['integral'] = max(-500, min(500, pid_state['integral']))
                
                I = KI * pid_state['integral']
                D = (KD * (error - pid_state['last_error']) / dt) if dt > 0.01 else 0.0
                pid_state['last_error'] = error
                pid_out = P + I + D
            else:
                pid_state['last_time'] = time.monotonic()

            if follow_colour == "Red":
                if lane_memory is None:
                    cx_blk = X_CENTRE  
                    if cnt_black is not None and area_black > 1000:
                        M_blk = cv.moments(cnt_black)
                        if M_blk['m00'] != 0: cx_blk = int(M_blk['m10'] / M_blk['m00'])
                    if cx < cx_blk: red_left_votes += 1
                    else: red_right_votes += 1
                    if red_left_votes > 3: lane_memory = "Left"
                    elif red_right_votes >= 4: lane_memory = "Right"
                was_on_red = True
            elif follow_colour == "Black" and was_on_red:
                if lane_memory == "Left": out_turn_cmd.value = 1  
                elif lane_memory == "Right": out_turn_cmd.value = 2  
                was_on_red, lane_memory, red_left_votes, red_right_votes = False, None, 0, 0
            else:
                was_on_red = False
                    
            cv.circle(crop_bgr, (X_CENTRE, Y_CENTRE), 5, (0, 255, 255), -1)
            cv.circle(crop_bgr, (cx, cy), 5, (0, 0, 255), -1)
            cv.putText(crop_bgr, f"FPS: {int(smoothed_fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            with disp_lock: np.copyto(disp_buf, crop_bgr)

            out_pid.value = pid_out
            out_cx.value = cx
            out_cy.value = cy
            out_is_priority.value = (follow_colour in ["Red", "Yellow"])
            out_lineArea.value = current_area 
            out_has_line.value = has_line

    except Exception as e:
        print(f"\n[line_worker] CRASHED: {e}\n", flush=True)
        traceback.print_exc()