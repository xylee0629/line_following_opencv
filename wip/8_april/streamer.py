import cv2
import threading
import time
from flask import Flask, Response
import config 

class WebStreamer:
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer 
        self.app = Flask(__name__)
        
        @self.app.route('/')
        def video_feed():
            return Response(self._generate_frames(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate_frames(self):
        import numpy as np 
        
        blank_frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank_frame, "Waiting for Camera...", (20, int(config.FRAME_HEIGHT/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        _, blank_buffer = cv2.imencode('.jpg', blank_frame)
        blank_bytes = blank_buffer.tobytes()

        while True:
            frame = self.frame_buffer.read()
            
            if frame is not None:
                display_frame = frame.copy()
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                with self.frame_buffer.lock:
                    current_cx = getattr(self.frame_buffer, 'cx', config.FRAME_CENTRE)
                    active_color = getattr(self.frame_buffer, 'color_tracked', "SEARCHING")
                
                # Inside _generate_frames() in streamer.py
                center_x = int(config.FRAME_CENTRE)
                
                # --- MATCH THE NEW MATH HERE ---
                scan_y = int(config.FRAME_HEIGHT * 0.70)
                
                # Draw the thicker 40px "Laser Slice" ROI boundaries
                cv2.line(display_frame, (0, scan_y), (config.FRAME_WIDTH, scan_y), (0, 255, 255), 1)
                cv2.line(display_frame, (0, scan_y+40), (config.FRAME_WIDTH, scan_y+40), (0, 255, 255), 1)
                # -------------------------------
                
                cv2.line(display_frame, (center_x, 0), (center_x, config.FRAME_HEIGHT), (0, 255, 0), 1)
                ui_colors = {
                    "RED": (0, 0, 255),
                    "YELLOW": (0, 255, 255),
                    "BLACK": (50, 50, 50), 
                    "SEARCHING": (255, 255, 255)
                }
                dot_color = ui_colors.get(active_color, (255, 255, 255))
                
                if current_cx is not None:
                    draw_cx = int(current_cx)
                    y_position = scan_y + 10 
                    cv2.circle(display_frame, (draw_cx, y_position), 8, dot_color, -1)
                    cv2.line(display_frame, (center_x, y_position), (draw_cx, y_position), (255, 0, 0), 2)

                cv2.putText(display_frame, f"TRACKING: {active_color}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, dot_color, 2)
                            
                error_val = abs(current_cx - config.FRAME_CENTRE) if current_cx else 0
                cv2.putText(display_frame, f"CX: {current_cx} | ERR: {error_val:.1f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
                           
            time.sleep(0.066)

    def start(self, port=5000):
        threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False),
            daemon=True
        ).start()
        print(f"[SYSTEM] Web Streamer running at http://<robot_ip>:{port}")