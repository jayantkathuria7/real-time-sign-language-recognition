import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import os
import json
from playsound import playsound

# Font paths - keeping these from your original code
font_paths = {
    'hi': 'fonts/NotoSansDevanagari-Regular.ttf',  # Hindi
    'gu': 'fonts/NotoSansGujarati-Regular.ttf',     # Gujarati
    'pa': 'fonts/NotoSansGurmukhi-Regular.ttf',     # Punjabi
    'ur': 'fonts/NotoNastaliqUrdu-Regular.ttf'      # Urdu
}

# Load translations
with open("translations/translations.json", "r", encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)

def get_translation(word, language='hindi'):
    """Get translation for a word in the specified language."""
    if language in TRANSLATIONS and word in TRANSLATIONS[language]:
        return TRANSLATIONS[language][word]
    return word

def play_translation(word, language='hindi'):
    """Play the pre-generated audio for a word in the specified language."""
    # Fix path issues by normalizing path with proper forward slashes
    audio_path = os.path.join("audio", language, f"{word}.mp3")
    audio_path = os.path.normpath(audio_path).replace('\\', '/')
    
    if os.path.exists(audio_path):
        try:
            playsound(audio_path, block=False)
            return True
        except Exception as e:
            print(f"Error playing sound: {e}")
    else:
        print(f"Audio file not found: {audio_path}")
    return False

# Language codes mapping
language_codes = {
    'hindi': 'hi',
    'gujarati': 'gu', 
    'punjabi': 'pa',
    'urdu': 'ur'
}

# Load model and label encoder
model = tf.keras.models.load_model("models/cnn_lstm_model.keras")
with open("data/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils

# Constants
SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)
last_prediction = ""
last_translated = ""
selected_language = 'hindi'  # Change to 'gujarati', 'punjabi', 'urdu'

# Keypoint utils
def extract_hand_keypoints(results):
    lh = np.zeros((21, 3))
    rh = np.zeros((21, 3))
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
    return np.concatenate([lh, rh], axis=0)

def normalize_keypoints(keypoints):
    wrist = keypoints[0]
    keypoints -= wrist
    scale = np.linalg.norm(keypoints[4] - keypoints[20])
    return keypoints / (scale + 1e-6)

# Webcam loop
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    keypoints = extract_hand_keypoints(results)
    normalized = normalize_keypoints(keypoints)
    sequence.append(normalized)

    # Draw landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Prediction logic
    if len(sequence) == SEQUENCE_LENGTH:
        input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 126)
        prediction = model.predict(input_data)
        pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)
        print(pred_label, confidence)

        if confidence > 0.5:
            if pred_label != last_prediction:
                # Get pre-generated translation instead of API call
                translated_text = get_translation(pred_label, selected_language)
                last_prediction = pred_label
                last_translated = translated_text

                # Play pre-generated audio instead of API call
                play_translation(pred_label, selected_language)

            # Draw prediction and translated text using PIL
            pil_image = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_image)

            # Load correct font
            lang_code = language_codes.get(selected_language)
            font_path = font_paths.get(lang_code)
            font = ImageFont.truetype(font_path, 48)

            # Display English label and confidence
            draw.text((10, 30), f"Sign: {pred_label}", font=ImageFont.truetype("arial.ttf", 32), fill=(255, 255, 255))
            draw.text((10, 75), f"Conf: {confidence:.2f}", font=ImageFont.truetype("arial.ttf", 28), fill=(0, 255, 0))

            # Display translated label
            draw.text((10, 130), last_translated, font=font, fill=(255, 255, 255))

            # Convert back to OpenCV format
            frame = np.array(pil_image)

    # Show final frame
    cv2.imshow("Sign Language Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()