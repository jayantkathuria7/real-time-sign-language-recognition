import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
VIDEO_PATH = "data/videos"
WORDS = os.listdir(VIDEO_PATH)
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 42  

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

# Utility functions
def extract_hand_keypoints(results):
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    return np.concatenate([lh, rh], axis=0)

def normalize_keypoints(keypoints):
    wrist = keypoints[0]  # right wrist as origin
    keypoints -= wrist
    scale = np.linalg.norm(keypoints[4] - keypoints[20])  # thumb tip to pinky tip
    return keypoints / (scale + 1e-6)

def read_video(file):
    print(f"Reading File: {file}")
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

def process_all_videos():
    print("Processing All videos")
    sequences, labels = [], []
    for idx, word in enumerate(WORDS):
        word_path = os.path.join(VIDEO_PATH, word)
        for video_file in os.listdir(word_path):
            full_path = os.path.join(word_path, video_file)
            sequence = read_video(full_path)
            
            seq_len = len(sequence)
            
            # Check if the sequence length is less than the required length (SEQUENCE_LENGTH)
            if seq_len < SEQUENCE_LENGTH:
                # Pad the sequence with zeros to match the required sequence length
                pad = [np.zeros((42, 3))] * (SEQUENCE_LENGTH - seq_len)
                sequence += pad
                sequences.append(np.array(sequence))
                labels.append(idx)
            else:
                # Calculate the middle part of the sequence and extract SEQUENCE_LENGTH frames centered around it
                middle_index = seq_len // 2
                start_index = middle_index - (SEQUENCE_LENGTH // 2)
                end_index = middle_index + (SEQUENCE_LENGTH // 2)

                # Make sure the indices don't go out of bounds
                start_index = max(start_index, 0)
                end_index = min(end_index, seq_len)

                # Extract the middle frames
                middle_frames = sequence[start_index:end_index]

                # If we haven't reached SEQUENCE_LENGTH frames, pad the sequence
                if len(middle_frames) < SEQUENCE_LENGTH:
                    pad = [np.zeros((42, 3))] * (SEQUENCE_LENGTH - len(middle_frames))
                    middle_frames += pad

                sequences.append(np.array(middle_frames))
                labels.append(idx)
    
    return np.array(sequences), np.array(labels)


X_raw, y = process_all_videos()
np.save('data/X_raw.npy', X_raw)
np.save('data/y.npy', y)
print(X_raw.shape)
print(y.shape)
