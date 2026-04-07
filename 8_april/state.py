import threading 

class SharedState:
    def __init__(self):
        self.lock = threading.lock()
        self.steering = (0, 0)
        self.latest_frame = None
        
    def read(self):
        if self.latest_frame is not None:
            return self.latest_frame.copy()
        else:
            return None 
        
        