import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Utility functions
def extract_hand_keypoints(results):
    """Extract and combine left and right hand landmarks from MediaPipe results."""
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    return np.concatenate([lh, rh], axis=0)

def normalize_keypoints(keypoints):
    """Normalize hand landmarks relative to the wrist and hand size."""
    wrist = keypoints[0]  # left wrist as origin
    keypoints -= wrist
    scale = np.linalg.norm(keypoints[4] - keypoints[20])  # thumb tip to pinky tip
    return keypoints / (scale + 1e-6)

def read_video(file):
    """Read a video and extract normalized hand keypoints from each frame."""
    cap = cv2.VideoCapture(file)
    sequence = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        hand_keypoints = extract_hand_keypoints(results)
        normalized = normalize_keypoints(hand_keypoints)
        sequence.append(normalized)
    cap.release()
    return sequence