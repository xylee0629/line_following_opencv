import numpy as np 
    
# upgraded to 16:9 Widescreen
FRAME_WIDTH = 640  
FRAME_HEIGHT = 360 
FRAME_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)
FRAME_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3

# Dynamically calculate the bottom half of the screen for the line tracker
LINE_CROP_START = int(FRAME_HEIGHT / 2)
LINE_DISPLAY_HEIGHT = FRAME_HEIGHT - LINE_CROP_START

LINE_DISPLAY_SHAPE = (LINE_DISPLAY_HEIGHT, FRAME_WIDTH, 3)
LINE_DISPLAY_BYTES = LINE_DISPLAY_HEIGHT * FRAME_WIDTH * 3

IMG_DISPLAY_SHAPE = (FRAME_HEIGHT, FRAME_WIDTH, 3)
IMG_DISPLAY_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3

MOTOR_LEFT_NORMAL, MOTOR_RIGHT_NORMAL = 30, 30
MOTOR_LEFT_PUSH, MOTOR_RIGHT_PUSH = 33, 33

KP, KI, KD = 0.325, 0.0001, 0.015
FREQUENCY = 300
X_CENTRE = int(FRAME_WIDTH / 2)
Y_CENTRE = int(FRAME_HEIGHT/ 2)
DEBOUNCE_FRAMES = 2

SYMBOL_DICT = {
    0: (["latest/pushButton-1.jpg",    "latest/pushButton-2.jpg",    "latest/pushButton-3.jpg"],    20),
    1: (["latest/fingerPrint-1.jpg",   "latest/fingerPrint-2.jpg",   "latest/fingerPrint-3.jpg"],   18),
    2: (["latest/qrCode-1.jpg",        "latest/qrCode-2.jpg",        "latest/qrCode-3.jpg"],        15),
    3: (["latest/recycleSymbol.jpg"], 25),
    4: (["latest/hazardSymbol-1.jpg",  "latest/hazardSymbol-2.jpg",  "latest/hazardSymbol-3.jpg"],  18) # Lowered for motion blur
}

SYMBOL_NAME = {
    0: "pushButton",
    1: "fingerPrint",
    2: "qrCode",
    3: "recycleSymbol",
    4: "hazardSymbol",
}

IMAGE_COLOUR_RANGES = {
    "Green":    {"space": "HSV", "lower": np.array([40,  60,  50]), "upper": np.array([85,  255, 255])}, 
    "Yellow":   {"space": "HSV", "lower": np.array([25, 150,  50]), "upper": np.array([35,  255, 255])}, 
    "Purple":   {"space": "LAB", "lower": np.array([0, 145,  60 ]), "upper": np.array([255, 195, 135])}, 
    "Blue/Teal":{"space": "LAB", "lower": np.array([0 , 100,  60]), "upper": np.array([230, 165, 120])},
    "Red":      {"space": "LAB", "lower": np.array([0 , 160, 130]), "upper": np.array([255, 255, 180])}, 
    "Orange":   {"space": "LAB", "lower": np.array([0, 130, 165 ]), "upper": np.array([255, 180, 200])},
}

LINE_COLOUR_RANGES = {
    "Red":      {"lower_1": np.array([0, 100, 100]), "lower_2": np.array([160, 100, 100]), "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow":   {"lower": np.array([20, 80, 80]), "upper": np.array([40, 255, 255])},
    "Black":   {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 70])}
}

LABEL_TO_INSTRUCTION = {
    "Arrow (Left)":    "TURN_LEFT",
    "Arrow (Right)":   "TURN_RIGHT",
    "Arrow (Up)":      "MOVE_FORWARD",
    "pushButton":    "STOP",
    "hazardSymbol":  "STOP",
    "recycleSymbol": "360-TURN",
}



