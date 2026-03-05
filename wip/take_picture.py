from picamera2 import Picamera2
import cv2

# Initialize and configure the camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the camera pipeline
picam2.start()

print("Press 'c' to capture an image, or 'q' to quit.")

while True:
    # Grab the current frame from the camera as a NumPy array
    frame = picam2.capture_array()
    
    # Display the frame in an OpenCV window
    cv2.imshow("Live Preview", frame)
    
    # Capture keypress (wait 1 millisecond)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c'):
        # Capture the high-quality still
        picam2.capture_file("image.jpg")
        print("Image saved as 'image.jpg'!")
        break
    elif key == ord('q'):
        print("Exiting...")
        break

# Clean up resources
cv2.destroyAllWindows()
picam2.stop()
