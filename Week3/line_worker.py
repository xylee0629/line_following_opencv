import cv2
import numpy as np
import time
from multiprocessing import shared_memory
import traceback

from config import *
from vision_utils import bestContour

def line_worker(shm_name, frame_lock, line_ready_event, out_pid, out_reset_pid, out_cx, out_cy, out_turn_cmd, out_is_priority):
    try: 
        shm = shared_memory.SharedMemory(name=shm_name)
        frame_bf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

        lane_memory = None
        left_red_votes, right_red_votes = 0, 0
        was_on_red = False
        was_on_yellow = False
        pid_state = {
                'last_error': 0.0,
                'integral': 0.0,
                'last_time': time.monotonic()
            }
        pid_out = 0.0
        cx, cy = X_CENTRE, 150 # cy value need to check

        def get_bbox(contour):
            if contour is not None:
                return cv2.boundingRect(contour)
            else:
                return (0, 0, 0, 0)
        
        while True:
            # Waits for the frame to clear
            line_ready_event.wait()
            line_ready_event.clear()

            if out_reset_pid.value == True:
                pid_state['last_error'] = 0.0
                pid_state['integral'] = 0.0
                pid_state['last_time'] = time.monotonic()
                out_reset_pid.value = False
            
            with frame_lock: 
                frame = frame_bf.copy()

            # cap.read() reads in BGR format, need to convert from BGR2HSV
            crop_bgr = frame[FRAME_HEIGHT/2:FRAME_HEIGHT, :]
            crop_bgr_copy = crop_bgr.copy()
            crop_blur = cv2.GaussianBlur(crop_bgr, (3,3), 0)
            crop_hsv = cv2.cvtColor(crop_blur, cv2.COLOR_BGR2HSV)

            active_contour = None
            active_bbox = (0, 0, 0, 0)

            mask_red = cv2.bitwise_or(cv2.inRange(crop_hsv, LINE_COLOUR_RANGES["Red"]["lower_1"], LINE_COLOUR_RANGES["Red"]["upper_1"]), 
                                      cv2.inRange(crop_hsv, LINE_COLOUR_RANGES["Red"]["lower_2"], LINE_COLOUR_RANGES["Red"]["upper_2"]))
            mask_yellow = cv2.inRange(crop_hsv, LINE_COLOUR_RANGES["Yellow"]["lower"], LINE_COLOUR_RANGES["Yellow"]["upper"])
            mask_black = cv2.inRange(crop_hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])

            cnt_red, area_red = bestContour(mask_red)
            cnt_yellow, area_yellow = bestContour(mask_yellow)
            cnt_black, area_black = bestContour(mask_black)

            xr, yr, wr, hr = get_bbox(cnt_red)
            xy, yy, wy, hy = get_bbox(cnt_yellow)
            xb, yb, wb, hb = get_bbox(cnt_black)

            # check if the area is large enough, width and height of the box is more than minimum
            if area_red > 4500 and wr > 135 and hr > 20:
                active_contour = cnt_red
                active_bbox = (xr, yr, wr, hr)
                follow_colour = "Red"
                draw_colour = (0, 0, 255)
            elif area_yellow > 4500 and wy > 135 and hy > 20:
                active_contour = cnt_yellow
                active_bbox = (xy, yy, wy, hy)
                follow_colour = "Yellow"
                draw_colour = (0, 255, 255)
            elif area_black > 4500 and wb > 135 and hb > 20:
                active_contour = cnt_black
                active_bbox = (xb, yb, wb, hb)
                follow_colour = "Black"
                draw_colour = (0, 255, 0)
            else:
                active_contour = None
                follow_colour = "None"
                draw_colour = (0, 0, 0)

            has_line = False
            current_area = 0.0
            if active_contour is not None:
                has_line = True
                current_area = cv2.contourArea(active_contour)
                x, y, w, h = active_bbox
                
                M = cv2.moments(active_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                else:
                    cx, cy = x + (w // 2), y + (h // 2)

                # Parameters: image to draw on, input contour, draw all contours, colour to use, thickness of line
                cv2.drawContours(crop_bgr_copy, [active_contour], -1, draw_colour, 2)
                
                # PID using the centre found from the contour
                error = X_CENTRE - cx
                now = time.monotonic()
                dt = now - pid_state["last_time"]
                pid_state["last_time"] = now

                P = Kp * error
                pid_state["integral"] += error * dt
                pid_state['integral'] = max(-500, min(500, pid_state['integral']))
                
                I = Ki * pid_state['integral']
                D = (Kd * (error - pid_state['last_error']) / dt) if dt > 0.01 else 0.0
                pid_state['last_error'] = error
                pid_out = P + I + D
            else:
                pid_state["last_time"] = time.monotonic()


            # Line memory system
            if follow_colour == "Red":
                if lane_memory is not None:
                    cx_black = X_CENTRE
                    if cnt_black is not None and area_black > 1000:
                        M_black = cv2.moments(cnt_black)
                        if M_black['m00'] != 0:
                            cx_black = int(M_black['m10'] / M_black['m00'])

                    # Checks the number of red contour detected for a certain number of frames to prevent noise
                    if cx > cx_black:
                        left_red_votes += 1
                    else:
                        right_red_votes += 1
                    if left_red_votes > 5:
                        lane_memory = "Left"
                    elif right_red_votes > 5:
                        lane_memory = "Right"
                    elif left_red_votes > 5 and right_red_votes > 5:
                        lane_memory = "None"
                was_on_red = True
            elif follow_colour == "Black":
                if was_on_red:
                    if lane_memory == "Left":
                        out_turn_cmd.value = 1
                    elif lane_memory == "Right":
                        out_turn_cmd.value = 2

                # Reset all variables after following black
                was_on_red = False
                lane_memory = None
                left_red_votes, right_red_votes = 0,0
            else:
                was_on_red = False
                        
            out_pid.value = pid_out
            out_cx.value = cx
            out_cy.value = cy
            out_is_priority.value = (follow_colour in ["Red", "Yellow"]) # outputs True if the follower colour is Red or Yellow

    except Exception as e:
        traceback.print_exc()