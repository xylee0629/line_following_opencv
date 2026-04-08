import configparser
import os

config_parser = configparser.ConfigParser()
config_parser.optionxform = str  # Preserve case sensitivity

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config_parser.read(config_path)

# ==========================================
# CAMERA CONFIG
# ==========================================
FRAME_WIDTH = config_parser.getint('Camera', 'frame_width', fallback=640)
FRAME_HEIGHT = config_parser.getint('Camera', 'frame_height', fallback=480)
FRAME_CENTRE = int(config_parser.getfloat('Camera', 'frame_centre', fallback=FRAME_WIDTH / 2))
CAMERA_ID = config_parser.getint('Camera', 'camera_id', fallback=0)

# ==========================================
# MOTOR CONFIG
# ==========================================
FREQUENCY = config_parser.getint('Motor', 'frequency', fallback=100)
DUTY_CYCLE = config_parser.getfloat('Motor', 'duty_cycle', fallback=0.5)
KP = config_parser.getfloat('Motor', 'Kp', fallback=0.1)
KD = config_parser.getfloat('Motor', 'Kd', fallback=0.01)

# ==========================================
# PATHS
# ==========================================
# Path to the arrow template
ARROW_PATH = config_parser.get('Path', 'Arrow', fallback='assets/arrows/arrow_template.png')

# List of paths for symbols (symbols should be under the 'Path' section in the config.ini)
SYMBOL_PATHS = []
for name, filepath in config_parser.items('Path'):
    if name != "Arrow":
        SYMBOL_PATHS.append({"name": name, "filepath": filepath})

# ==========================================
# VIDEO CONFIG (for video source)
# ==========================================
VIDEO_SOURCE = config_parser.getint('Video', 'video_source', fallback=0)  # 0 for webcam, or video file path
FRAME_BUFFER_SIZE = config_parser.getint('Video', 'frame_buffer_size', fallback=5)

# ==========================================
# FILTER CONFIG (for motion blur handling)
# ==========================================
FILTER_ALPHA = config_parser.getfloat('Filter', 'filter_alpha', fallback=0.8)  # Frame averaging alpha