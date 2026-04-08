import threading 

class SharedState:
    def __init__(self):
        self.lock = threading.Lock() # Fixed capitalization
        self.steering = (0, 0)
        self.latest_frame = None
        
