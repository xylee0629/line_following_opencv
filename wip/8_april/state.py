import threading 
import config

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() 
        self.steering = (0, 0)
        self.latest_frame = None
        self.cx = config.FRAME_CENTRE 
        
        # Color Tracking & Junction Memory
        self.color_tracked = "SEARCHING" 
        self.path_count = 0
        self.pending_turn = None # e.g., "ARROW_LEFT" or "ARROW_RIGHT"
        
    def read(self):
        """Used by WebStreamer to safely fetch the latest frame"""
        with self.lock:
            return self.latest_frame