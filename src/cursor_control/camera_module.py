import cv2

# Function to initialize camera for capturing video
def initialize_camera():
    cam = cv2.VideoCapture(0)

     # Check if camera is successfully opened
    return cam if cam.isOpened() else None

# Function to read a frame from the camera
def read_frame(cam):
    _, frame = cam.read()
    return frame

# Function to flip the frame horizontally
def flip_frame(frame):
    return cv2.flip(frame, 1)

# Release the camera when video capturing is done
def release_camera(cam):
    if cam is not None:
        cam.release()