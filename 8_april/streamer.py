# streamer.py
import cv2
import threading
import time
from flask import Flask, Response

class WebStreamer:
    def __init__(self, frame_buffer):
        self.frame_buffer = frame_buffer
        self.app = Flask(__name__)
        
        # Define the web route
        @self.app.route('/')
        def video_feed():
            return Response(self._generate_frames(), 
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def _generate_frames(self):
        """Continuously pulls frames from the buffer and encodes them to JPEG."""
        while True:
            frame = self.frame_buffer.read()
            if frame is not None:
                # Compress the frame to JPEG to send over Wi-Fi quickly
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Yield the frame in byte format for the web browser
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.066) # Limit to ~15 FPS to save CPU

    def start(self, port=5000):
        """Starts the Flask server in a background thread."""
        # Flask is blocking, so we MUST run it in a daemon thread
        threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False),
            daemon=True
        ).start()
        print(f"[SYSTEM] Web Streamer running at http://<robot_ip>:{port}")