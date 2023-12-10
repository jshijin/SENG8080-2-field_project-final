from camera_module import initialize_camera, read_frame, flip_frame, release_camera
from face_module import process_face
import pyautogui
import mediapipe as mp
import cv2
import time

def main():
    # Camera and Face mesh initialization
    cam = initialize_camera()
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    screen_w, screen_h = pyautogui.size()
    
    # Initialization for Cursor Movement
    count = True

    while True:
        # Read a frame from the camera
        frame = read_frame(cam)
        if frame is None:
            print("Error: Unable to read frame from the camera.")
            break
        # Flip the frame horizontally for a mirrored view
        frame = flip_frame(frame)


    release_camera(cam)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()