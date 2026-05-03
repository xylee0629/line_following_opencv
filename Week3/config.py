import numpy as np

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)
FRAME_BYTES  = FRAME_WIDTH * FRAME_HEIGHT * 3

X_CENTRE = int(FRAME_WIDTH / 2)
Y_CENTRE = int(FRAME_HEIGHT / 2)

Kp = 0.0
Kd = 0.0
Ki = 0.0
FREQUENCY = 300

MOTOR_BASE_SPEED = 30
MOTOR_PUSH_SPEED = 33

DEBOUNCE_FRAMES = 3

LINE_COLOUR_RANGES = {
    "Red":      {"lower_1": np.array([0, 100, 100]), "lower_2": np.array([160, 100, 100]), "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow":   {"lower": np.array([20, 80, 80]), "upper": np.array([40, 255, 255])},
    "Black":   {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 70])}
}

IMAGE_COLOUR_RANGES = {
    "Green":    {"space": "HSV", "lower": np.array([40,  60,  50]), "upper": np.array([85,  255, 255])}, 
    "Yellow":   {"space": "HSV", "lower": np.array([25, 150,  50]), "upper": np.array([35,  255, 255])}, 
    "Purple":   {"space": "LAB", "lower": np.array([0, 145,  60 ]), "upper": np.array([255, 195, 135])}, 
    "Blue/Teal":{"space": "LAB", "lower": np.array([0 , 100,  60]), "upper": np.array([230, 165, 120])},
    "Red":      {"space": "LAB", "lower": np.array([0 , 160, 130]), "upper": np.array([255, 255, 180])}, 
    "Orange":   {"space": "LAB", "lower": np.array([0, 130, 165 ]), "upper": np.array([255, 180, 200])},
}

SYMBOL_DICT = {
    0: (["/home/raspberrypi/line_following_opencv/Week3/images/fingerPrint-1.jpg",    "/home/raspberrypi/line_following_opencv/Week3/images/pushButton-2.jpg",    "/home/raspberrypi/line_following_opencv/Week3/images/pushButton-3.jpg"],    35),
    1: (["/home/raspberrypi/line_following_opencv/Week3/images/fingerPrint-1.jpg",   "/home/raspberrypi/line_following_opencv/Week3/images/fingerPrint-2.jpg",   "/home/raspberrypi/line_following_opencv/Week3/images/fingerPrint-3.jpg"],   30),
    2: (["/home/raspberrypi/line_following_opencv/Week3/images/qrCode-1.jpg",        "/home/raspberrypi/line_following_opencv/Week3/images/qrCode-2.jpg",        "/home/raspberrypi/line_following_opencv/Week3/images/qrCode-3.jpg"],        25),
    3: (["/home/raspberrypi/line_following_opencv/Week3/images/recycleSymbol.jpg"], 25),
    4: (["/home/raspberrypi/line_following_opencv/Week3/images/hazardSymbol-1.jpg",  "/home/raspberrypi/line_following_opencv/Week3/images/hazardSymbol-2.jpg",  "/home/raspberrypi/line_following_opencv/Week3/images/hazardSymbol-3.jpg"],  30) # Lowered for motion blur
}

SYMBOL_NAME = {
    0: "pushButton",
    1: "fingerPrint",
    2: "qrCode",
    3: "recycleSymbol",
    4: "hazardSymbol",
}

LABEL_TO_INSTRUCTION = {
    "Arrow (Left)":    "TURN_LEFT",
    "Arrow (Right)":   "TURN_RIGHT",
    "Arrow (Up)":      "MOVE_FORWARD",
    "pushButton":    "STOP",
    "hazardSymbol":  "STOP",
    "recycleSymbol": "360-TURN",
}