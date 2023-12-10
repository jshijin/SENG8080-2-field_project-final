from camera_module import initialize_camera, read_frame, flip_frame, release_camera
from face_module import process_face, draw_landmarks, move_cursor
from control_module import double_click, process_blink_action
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

        landmark_points, frame_w, frame_h = process_face(frame, face_mesh)

        if count:
            if landmark_points:
                landmarks = landmark_points[0].landmark
                draw_landmarks(frame, landmarks, frame_w, frame_h)
                x = int(landmarks[475].x * frame_w)
                y = int(landmarks[475].y * frame_h)

                # Check if the cursor moves correctly
                move_cursor(frame, x, y)

                left = [landmarks[153], landmarks[158]]
                process_blink_action(frame, left, double_click, 0.006, frame_w, frame_h)


    release_camera(cam)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()