import configparser, os

config = configparser.ConfigParser()
config.optionxform = str

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# Cast to correct type
FRAME_WIDTH = config.getint('Camera' , 'frame_width')
FRAME_HEIGHT = config.getint('Camera', 'frame_height')
FRAME_CENTRE = config.getfloat('Camera', 'frame_centre')

FREQUENCY = config.getint('Motor', 'frequency')
DUTY_CYCLE = config.getfloat('Motor', 'duty_cycle')
KP = config.getfloat('Motor', 'kp')
KD = config.getfloat('Motor', 'kd')

ARROW_PATH = config.get('Path', 'Arrow')

SYMBOL_PATHS = []
for name, filepath in config.items('Path'):
    if name != "Arrow":
        SYMBOL_PATHS.append({
            "name": name,
            "filepath": filepath
        })
