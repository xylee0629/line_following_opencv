import cv2
import numpy as np
import config

class Vision:
    def __init__(self):
        self.pixel_threshold = 3000
        self.bounds = {
            "RED1": (np.array(config.RED1_LOWER), np.array(config.RED1_UPPER)),
            "RED2": (np.array(config.RED2_LOWER), np.array(config.RED2_UPPER)),
            "YELLOW": (np.array(config.YELLOW_LOWER), np.array(config.YELLOW_UPPER)),
            "BLACK": (np.array(config.BLACK_LOWER), np.array(config.BLACK_UPPER))
        }
        
    def line_processing(self, frame, symbol=None):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        masks = {
            "RED": cv2.bitwise_or(cv2.inRange(hsv_frame, *self.bounds["RED1"]),
                                  cv2.inRange(hsv_frame, *self.bounds["RED2"])),
            "YELLOW": cv2.inRange(hsv_frame, *self.bounds["YELLOW"]),
            "BLACK": cv2.inRange(hsv_frame, *self.bounds["BLACK"])
        }
        
        combined_mask = cv2.bitwise_or(masks["BLACK"], cv2.bitwise_or(masks["YELLOW"], masks["RED"]))
        
        height = config.FRAME_HEIGHT
        width = config.FRAME_WIDTH
        y_centre = int(height * 0.6)
        x_centre_start = int(width * 0.4)
        x_centre_end = int(width * 0.8)
        
        centre_box = combined_mask[y_centre:height, x_centre_start:x_centre_end]
        cx = config.FRAME_CENTRE
        M = cv2.moments(centre_box)
        if M["m00"] > 0:
            box_cx = int(M["m10"] / M["m00"])
            cx = box_cx + x_centre_start
            
            
        j_type_overall = "NONE"
        cx_left_final, cx_right_final = None, None
        left_color, right_color = "NONE", "NONE"


        if symbol in ["LEFT", "RIGHT"]:
            # FAST PATH: We already know where we are going.
            # Just scan the combined mask once to find the physical path coordinates.
            j_type_overall, cx_left_final, cx_right_final = self.junction_scan(combined_mask, height, width)
            
            # We don't care about the color, so we leave them as "NONE"
            
        else:
            # SLOW PATH: No symbol detected. We must map out the colors to make a decision.
            for color in ["RED", "YELLOW", "BLACK"]:
                j_type, cx_l, cx_r = self.junction_scan(masks[color], height, width)
                
                if j_type in ["LEFT", "T_OR_Y_BOTH"] and left_color == "NONE":
                    left_color = color
                    cx_left_final = cx_l
                    if j_type_overall in ["NONE", "RIGHT"]:
                        j_type_overall = "LEFT" if j_type_overall == "NONE" else "T_OR_Y_BOTH"
                        
                if j_type in ["RIGHT", "T_OR_Y_BOTH"] and right_color == "NONE":
                    right_color = color
                    cx_right_final = cx_r
                    if j_type_overall in ["NONE", "LEFT"]:
                        j_type_overall = "RIGHT" if j_type_overall == "NONE" else "T_OR_Y_BOTH"

        # Send whatever we found to the brain
        return self.line_logic(cx, j_type_overall, cx_left_final, cx_right_final, left_color, right_color, symbol)
            
    
    def junction_scan(self, mask, height, width):
        y_slice = slice(0, int(height * 0.6))
        
        left_box = mask[y_slice, slice(0, int(width * 0.4))]
        left_px = cv2.countNonZero(left_box)
        
        right_box = mask[y_slice, slice(int(width * 0.8), width)]
        right_px = cv2.countNonZero(right_box)

        cx_left, cx_right = None, None

        M_left = cv2.moments(left_box)
        if M_left["m00"] > 0:
            cx_left = int(M_left["m10"] / M_left["m00"]) + 0
        
        M_right = cv2.moments(right_box)
        if M_right["m00"] > 0:
            # Fixed typos: changed M_left to M_right, and offset to int(width * 0.8)
            cx_right = int(M_right["m10"] / M_right["m00"]) + int(width * 0.8)
            
        if left_px > self.pixel_threshold and right_px > self.pixel_threshold:
            return "T_OR_Y_BOTH", cx_left, cx_right
        if left_px > self.pixel_threshold:
            return "LEFT", cx_left, None
        if right_px > self.pixel_threshold:
            return "RIGHT", None, cx_right
            
        return "NONE", None, None
        
    def line_logic(self, cx, j_type, cx_left, cx_right, left_color, right_color, symbol=None):
        # If is "NONE", "LEFT", or "RIGHT": 
            # straight line. return cx
        if j_type in ["NONE", "LEFT", "RIGHT"]:
            return cx
            
        # If is "T or Y_BOTH":
        if j_type == "T_OR_Y_BOTH":
            
            # if the symbol is "LEFT":
                # return cx_left
            if symbol == "LEFT" and cx_left is not None:
                return cx_left
                
            # if the symbol is "RIGHT":
                # return cx_right
            elif symbol == "RIGHT" and cx_right is not None:
                return cx_right
                
            # if there is no symbol:
            else:
                # Assign numerical weights to easily compare priority
                priority = {"RED": 3, "YELLOW": 2, "BLACK": 1, "NONE": 0}
                
                # If the colour detected is RED (or higher priority):
                    # return cx of the corresponding colour line
                if priority[left_color] > priority[right_color]:
                    return cx_left
                elif priority[right_color] > priority[left_color]:
                    return cx_right
                else:
                    # Tie-breaker if both sides are the exact same color
                    return cx_left 
                    
        return cx