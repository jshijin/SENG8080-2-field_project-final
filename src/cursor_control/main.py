from camera_module import initialize_camera, read_frame, flip_frame, release_camera
from face_module import process_face, draw_landmarks, move_cursor
from control_module import double_click, process_blink_action,right_click, move_cursor_to
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
                screen_x, screen_y = draw_landmarks(frame, landmarks, frame_w, frame_h, screen_w, screen_h)

                # Check if the cursor moves correctly
                move_cursor_to(screen_x, screen_y)

                left = [landmarks[153], landmarks[158]]
                process_blink_action(frame, left, double_click, 0.006, frame_w, frame_h)
                #Processing Right eye movement
                right = [landmarks[380], landmarks[385]]
                process_blink_action(frame, right, right_click, 0.004, frame_w, frame_h)
                
                # Processing Mouth landmarks
                mouth = [landmarks[15], landmarks[13]]
                for landmark in mouth:
                    x = int(landmark.x * frame_w)

                    y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 254))
                if mouth[0].y - mouth[1].y > 0.08:
                    count = False
                    print("a")
                    pyautogui.sleep(0.2)
                    time.sleep(2)
        else:
            if landmark_points:
                landmarks = landmark_points[0].landmark
                mouth = [landmarks[15], landmarks[13]]
                for landmark in mouth:
                    x = int(landmark.x * frame_w)

                    y = int(landmark.y * frame_h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 255))
                if (mouth[0].y - mouth[1].y) > 0.08:
                    count = True
                    print("b")
                    pyautogui.sleep(0.2)
                    time.sleep(1)

            # Handle the case when count is False
        cv2.imshow('Eye Control cursor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    release_camera(cam)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()