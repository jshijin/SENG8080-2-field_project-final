import pyautogui
import time
import cv2

# Function to process a blink action based on the positions of landmarks detected in a frame
def process_blink_action(frame, landmarks, action, threshold, frame_w, frame_h):
    for landmark in landmarks:
        x = int(landmark.x * frame_w)
        y = int(landmark.y * frame_h)
        cv2.circle(frame, (x, y), 3, (0, 255, 255))

    if landmarks[0].y - landmarks[1].y < threshold:
        action()
        pyautogui.sleep(0.2)

# Function to process double click
def double_click():
    pyautogui.doubleClick()
    pyautogui.sleep(0.2)
