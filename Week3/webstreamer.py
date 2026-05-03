from flask import Flask, Response, render_template_string
import cv2 
import numpy as np
import time
from multiprocessing import shared_memory

app = Flask(__name__)

# Global references set by the worker entry point
_line_shm_name = _line_shape = _line_lock = None
_img_shm_name  = _img_shape  = _img_lock  = None

# A simple, dark-mode HTML page to view both streams side-by-side
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Vision Stream</title>
    <style>
        body { background-color: #1e1e1e; color: white; font-family: sans-serif; text-align: center; }
        .container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px; }
        .stream-box { background: #2d2d2d; padding: 10px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.5); }
        img { border-radius: 4px; max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>🤖 Robot Live Feed</h1>
    <div class="container">
        <div class="stream-box">
            <h2>Image Recognition (Full)</h2>
            <img src="/stream_img" width="640">
        </div>
        <div class="stream-box">
            <h2>Line Tracking (Cropped)</h2>
            <img src="/stream_line" width="640">
        </div>
    </div>
</body>
</html>
"""

def generate_frames(shm_name, shape, lock):
    """Generator that reads shared memory, encodes to JPEG, and yields byte chunks."""
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    except Exception as e:
        print(f"[streamer] ERROR connecting to shm: {e}")
        return

    while True:
        with lock:
            frame = buf.copy()
        
        #Compress the image! (40% quality reduces file size by ~80%)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not ret:
            time.sleep(0.05)
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Cap the stream at ~15 FPS. 
        # This stops the network from flooding and causing browser lag.
        time.sleep(0.06)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/stream_line')
def stream_line():
    return Response(generate_frames(_line_shm_name, _line_shape, _line_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_img')
def stream_img():
    return Response(generate_frames(_img_shm_name, _img_shape, _img_lock),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_streamer(line_shm_name, line_shape, line_lock, img_shm_name, img_shape, img_lock, port=5000):
    global _line_shm_name, _line_shape, _line_lock
    global _img_shm_name, _img_shape, _img_lock
    
    _line_shm_name, _line_shape, _line_lock = line_shm_name, line_shape, line_lock
    _img_shm_name, _img_shape, _img_lock = img_shm_name, img_shape, img_lock
    
    # Disable default Flask request logging to keep your Pi console clean
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    print(f"[streamer] Web server starting. View stream at: http://<raspberry-pi-ip>:{port}")
    # Checks the entire local wifi for the ip address of the pi
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)