import cv2
import mediapipe as mp

#  Function to process the face landmarks
def process_face(frame, face_mesh):
    frame_h, frame_w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    return landmark_points, frame_w, frame_h

