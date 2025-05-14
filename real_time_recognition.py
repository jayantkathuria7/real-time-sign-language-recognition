import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
import time
from collections import deque
import matplotlib.pyplot as plt
import pickle
import json
from PIL import Image, ImageDraw, ImageFont 
import pygame  # For audio playback

# Font paths
font_paths = {
    'hi': 'fonts/NotoSansDevanagari-Regular.ttf',
    'gu': 'fonts/NotoSansGujarati-Regular.ttf',
    'pa': 'fonts/NotoSansGurmukhi-Regular.ttf',
    'ur': 'fonts/NotoNastaliqUrdu-Regular.ttf'
}

# Load translations
with open("translations/translations.json", "r", encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)

def get_translation(word, language='hindi'):
    """Get translation for a word in the selected language"""
    if language in TRANSLATIONS and word in TRANSLATIONS[language]:
        return TRANSLATIONS[language][word]
    return word

# Language mappings
language_codes = {
    'hindi': 'hi',
    'gujarati': 'gu',
    'punjabi': 'pa',
    'urdu': 'ur'
}
# Load your trained model
model = tf.keras.models.load_model('models/cnn_lstm_model.keras')

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Constants
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 42  # 21 points per hand, 2 hands
WORDS = os.listdir('data/videos3')
CONFIDENCE_THRESHOLD = 0.6

# Setup frame buffer
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

def plot_landmark_sequence(sequence, title="Landmark Trajectories"):
    sequence = sequence.reshape(30, 42, 3)
    fig = plt.figure(figsize=(12, 4))
    for i in range(42):
        x = sequence[:, i, 0]
        y = sequence[:, i, 1]
        plt.plot(x, label=f"L{i}_x", alpha=0.5)
        plt.plot(y, label=f"L{i}_y", alpha=0.5)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("Normalized Position")
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0), ncol=2, fontsize='xx-small')
    plt.tight_layout()
    plt.show()

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

def preprocess_for_model(sequence):
    sequence = np.array(sequence)
    sequence = sequence.reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 3)
    return sequence

def main():
    cap = cv2.VideoCapture(1)

    # Initialize pygame for audio
    pygame.mixer.init()
    
    prev_time = 0
    predictions_buffer = []
    smoothing_window = 5
    pred_every_n_frames = 2
    frame_count = 0

    current_prediction = "No hand detected"
    confidence = 0.0
    last_translated = ""
    
    # Audio playback control variables
    last_played_sign = ""
    last_played_time = 0
    audio_cooldown = 2.0  # seconds between audio playbacks
    stable_detection_frames = 0
    required_stable_frames = 10  # number of frames a sign must be stable to play audio
    auto_play_audio = False  # Flag to toggle automatic audio playback

    # Initialize language settings
    selected_language = 'hindi'

    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic_processor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_processor.process(frame_rgb)

            frame_with_landmarks = frame.copy()
            hands_detected = False

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                hands_detected = True
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                hands_detected = True

            if hands_detected:
                hand_keypoints = extract_hand_keypoints(results)
                normalized = normalize_keypoints(hand_keypoints)
                frame_buffer.append(normalized)

                if len(frame_buffer) == SEQUENCE_LENGTH and frame_count % pred_every_n_frames == 0:
                    input_sequence = preprocess_for_model(list(frame_buffer))
                    pred = model.predict(input_sequence, verbose=0)[0]
                    predicted_class = np.argmax(pred)
                    current_confidence = pred[predicted_class]

                    predictions_buffer.append((predicted_class, current_confidence))
                    if len(predictions_buffer) > smoothing_window:
                        predictions_buffer.pop(0)

                    class_counts = {}
                    confidence_sums = {}
                    for cls, conf in predictions_buffer:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        confidence_sums[cls] = confidence_sums.get(cls, 0) + conf

                    smooth_class = max(class_counts, key=class_counts.get)
                    avg_confidence = confidence_sums[smooth_class] / class_counts[smooth_class]

                    if avg_confidence > 0.5:
                        new_prediction = WORDS[smooth_class]
                        
                        # Check if this is a new or stable prediction
                        if new_prediction == current_prediction:
                            stable_detection_frames += 1
                        else:
                            stable_detection_frames = 0
                        
                        current_prediction = new_prediction
                        confidence = avg_confidence
                        # Get translation for the detected word
                        last_translated = get_translation(current_prediction, selected_language)
                        
                        # Play audio if the sign has been stable and enough time has passed
                        current_time = time.time()
                        if (auto_play_audio and 
                            stable_detection_frames >= required_stable_frames and 
                            (current_prediction != last_played_sign or 
                             current_time - last_played_time > audio_cooldown)):
                            
                            # Construct the audio file path
                            audio_file = f"audio/{selected_language}/{current_prediction}.mp3"
                            if os.path.exists(audio_file):
                                try:
                                    pygame.mixer.music.load(audio_file)
                                    pygame.mixer.music.play()
                                    last_played_sign = current_prediction
                                    last_played_time = current_time
                                    print(f"Playing audio for: {current_prediction}")
                                except Exception as e:
                                    print(f"Error playing audio: {e}")
                            else:
                                print(f"Audio file not found: {audio_file}")
                    else:
                        current_prediction = "No sign detected"
                        confidence = avg_confidence
                        last_translated = ""
                        stable_detection_frames = 0
            else:
                current_prediction = "No hand detected"
                confidence = 0.0
                last_translated = ""
                stable_detection_frames = 0

            frame_count += 1

            # Create overlay with PIL for better font rendering
            pil_image = Image.fromarray(cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # Load font for translation
            lang_code = language_codes.get(selected_language)
            font_path = font_paths.get(lang_code, 'arial.ttf')
            
            try:
                font = ImageFont.truetype(font_path, 40)
                eng_font = ImageFont.truetype("arial.ttf", 30)
            except IOError:
                # Fallback to default font
                font = ImageFont.load_default()
                eng_font = ImageFont.load_default()

            # Display FPS
            draw.text((10, 30), f"FPS: {fps:.1f}", font=eng_font, fill=(0, 255, 0))
            
            # Display text with status indicators
            draw.text((10, 70), f"Sign: {current_prediction}", font=eng_font, fill=(255, 0, 255))
            
            # Add audio status indicator
            audio_mode_status = "Auto" if auto_play_audio else "Manual"
            audio_ready_status = ""
            if stable_detection_frames > 0 and auto_play_audio:
                audio_ready_status = f" (Ready in {required_stable_frames - stable_detection_frames})" if stable_detection_frames < required_stable_frames else " (Ready)"
            
            # Display confidence if a sign is detected
            if current_prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
                draw.text((10, 110), f"Confidence: {confidence:.2f}", font=eng_font, fill=(255, 0, 0))
                draw.text((10, 230), f"Audio: {audio_mode_status}{audio_ready_status}", font=eng_font, fill=(0, 200, 200))
            
            # Display language name
            draw.text((10, 150), f"Language: {selected_language.capitalize()}", font=eng_font, fill=(255, 255, 255))
            
            # Display translation
            if last_translated and current_prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
                draw.text((10, 190), last_translated, font=font, fill=(255, 255, 0))
            
            # Convert back to OpenCV format
            frame_with_landmarks = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)    
            # cv2.putText(frame_with_landmarks, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame_with_landmarks, f"Sign: {current_prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # if current_prediction not in ["No hand detected", "No sign detected"]:
            #     cv2.putText(frame_with_landmarks, f"Confidence: {confidence:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Real-time Sign Detection', frame_with_landmarks)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                selected_language = 'hindi'
                print("Switched to Hindi")
                # Reset audio control variables on language change
                last_played_sign = ""
                stable_detection_frames = 0
            elif key == ord('g'):
                selected_language = 'gujarati'
                print("Switched to Gujarati")
                last_played_sign = ""
                stable_detection_frames = 0
            elif key == ord('p'):
                selected_language = 'punjabi'
                print("Switched to Punjabi")
                last_played_sign = ""
                stable_detection_frames = 0
            elif key == ord('u'):
                selected_language = 'urdu'
                print("Switched to Urdu")
                last_played_sign = ""
                stable_detection_frames = 0
            elif key == ord('a'):  # Toggle automatic audio playback
                auto_play_audio = not auto_play_audio
                print(f"Automatic audio playback: {'ON' if auto_play_audio else 'OFF'}")
                
            elif key == ord(' '):  # Spacebar to manually play audio for current prediction
                if current_prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
                    audio_file = f"audio/{selected_language}/{current_prediction}.mp3"
                    if os.path.exists(audio_file):
                        try:
                            pygame.mixer.music.load(audio_file)
                            pygame.mixer.music.play()
                            print(f"Manually playing audio for: {current_prediction}")
                        except Exception as e:
                            print(f"Error playing audio: {e}")
                    else:
                        print(f"Audio file not found: {audio_file}")

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # Clean up pygame resources

if __name__ == "__main__":
    main()