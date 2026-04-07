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
KP = config.getfloat('Motor', 'Kp')
KD = config.getfloat('Motor', 'Kd')

RED1_LOWER = [int(x) for x in config.get('Vision', 'red1_lower').split(',')]
RED1_UPPER = [int(x) for x in config.get('Vision', 'red1_upper').split(',')]
RED2_LOWER = [int(x) for x in config.get('Vision', 'red2_lower').split(',')]
RED2_UPPER = [int(x) for x in config.get('Vision', 'red2_upper').split(',')]
YELLOW_LOWER = [int(x) for x in config.get('Vision', 'yellow_lower').split(',')]
YELLOW_UPPER = [int(x) for x in config.get('Vision', 'yellow_upper').split(',')]
BLACK_LOWER = [int(x) for x in config.get('Vision', 'black_lower').split(',')]
BLACK_UPPER = [int(x) for x in config.get('Vision', 'black_upper').split(',')]

ARROW_PATH = config.get('Path', 'Arrow')

SYMBOL_PATHS = []
for name, filepath in config.items('Path'):
    if name != "Arrow":
        SYMBOL_PATHS.append({
            "name": name,
            "filepath": filepath
        })
