import cv2
import mediapipe as mp
import pyautogui

#  Function to process the face landmarks
def process_face(frame, face_mesh):
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    return landmark_points, frame_w, frame_h

# Fnction to draw landmarks on the input frame.
def draw_landmarks(frame, landmarks, frame_w, frame_h, screen_w, screen_h ):
    for id, landmark in enumerate(landmarks[474:478]):
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0))
        if id == 1:
                    screen_x = screen_w / frame_w * x
                    screen_y = screen_h / frame_h * y
                    #pyautogui.moveTo(screen_x, screen_y)
    return screen_x, screen_y
# Function to draw a curser on the input frame using specified coordinates
def move_cursor(frame, x, y):
    cv2.circle(frame, (x, y), 3, (0, 255, 0))
