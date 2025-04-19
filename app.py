import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
from collections import deque
import mediapipe as mp
import time
import os
from PIL import Image
import json

# Set page config
st.set_page_config(
    page_title="Sign Language Recognition",
    page_icon="👋",
    layout="wide"
)

# Load pre-generated translations
try:
    with open("translations/translations.json", "r", encoding="utf-8") as f:
        pre_translations = json.load(f)
    translation_module_available = True
except Exception as e:
    pre_translations = {}
    translation_module_available = False
    st.warning(f"Could not load pre-saved translations. Error: {e}")

# App title and description
st.title("Sign Language Recognition System")
st.markdown("""
This application recognizes sign language gestures in real-time using a trained deep learning model.
It can translate the recognized signs into different languages and play audio from pre-generated files.
""")

# Sidebar
st.sidebar.header("Settings")
camera_options = ["Default Camera (0)", "External Camera (1)"]
selected_camera = st.sidebar.selectbox("Select Camera", camera_options)
camera_id = 0 if "Default" in selected_camera else 1

languages = {
    "Hindi": "hindi",
    "Gujarati": "gujarati",
    "Punjabi": "punjabi",
    "Urdu": "urdu"
}
target_language = st.sidebar.selectbox("Translate To", list(languages.keys()))
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)

# Load model and label encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model("models/cnn_lstm_model.keras")
        with open("data/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return model, label_encoder, True
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        return None, None, False

model, label_encoder, model_loaded = load_model_and_encoder()

# Constants
SEQUENCE_LENGTH = 30

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

def convert_opencv_to_pil(opencv_image):
    color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_converted)
    return pil_image

# Main layout
col1, col2 = st.columns([2, 1])
with col1:
    video_placeholder = st.empty()
    status_text = st.empty()

with col2:
    st.subheader("Recognition Results")
    prediction_text = st.empty()
    confidence_text = st.empty()

    st.subheader("Translation")
    translation_placeholder = st.empty()
    translation_image = st.empty()
    audio_placeholder = st.empty()

# Control buttons
start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

if start_button:
    st.session_state.run_camera = True
if stop_button:
    st.session_state.run_camera = False

# Main logic
if st.session_state.run_camera and model_loaded:
    status_text.info("Starting camera... Please wait.")

    try:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error(f"Could not open camera {camera_id}.")
            st.session_state.run_camera = False
        else:
            status_text.success("Camera activated!")

            sequence = deque(maxlen=SEQUENCE_LENGTH)
            last_prediction = ""

            holistic = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Failed to receive frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                if show_landmarks:
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                try:
                    keypoints = extract_hand_keypoints(results)
                    normalized = normalize_keypoints(keypoints)
                    sequence.append(normalized)
                except Exception as e:
                    status_text.warning(f"Keypoint error: {e}")
                    continue

                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 126)
                    prediction = model.predict(input_data, verbose=0)
                    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                    confidence = np.max(prediction)

                    if confidence > confidence_threshold:
                        prediction_text.markdown(f"**Recognized Sign:** {pred_label}")
                        confidence_text.markdown(f"**Confidence:** {confidence:.2f}")

                        cv2.putText(frame, f"Sign: {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Translation + Audio
                        if pred_label != last_prediction and translation_module_available:
                            lang_key = languages[target_language].lower()
                            translated_text = pre_translations.get(lang_key, {}).get(pred_label, None)

                            if translated_text:
                                translation_placeholder.markdown(f"**Translation ({target_language}):** {translated_text}")
                                translation_image.empty()

                                audio_path = f"audio/{lang_key}/{pred_label}.mp3"
                                if os.path.exists(audio_path):
                                    with open(audio_path, "rb") as f:
                                        audio_bytes = f.read()
                                        audio_placeholder.audio(audio_bytes, format="audio/mp3")
                            else:
                                translation_placeholder.markdown("*No translation available for this sign.*")

                            last_prediction = pred_label
                    else:
                        prediction_text.markdown("**Recognized Sign:** Waiting for clear sign...")
                        confidence_text.markdown(f"**Confidence:** {confidence:.2f}")
                        translation_placeholder.empty()
                        audio_placeholder.empty()

                video_placeholder.image(convert_opencv_to_pil(frame), channels="RGB", use_column_width=True)
                time.sleep(0.01)

            cap.release()
            status_text.info("Camera stopped")

    except Exception as e:
        status_text.error(f"Error: {e}")
        st.session_state.run_camera = False

elif not model_loaded:
    st.error("Model or label encoder not loaded. Please check your files.")

if not st.session_state.run_camera:
    video_placeholder.info("Click 'Start Camera' to begin sign language recognition.")

    st.subheader("How to use this app")
    st.markdown("""
    1. Click the 'Start Camera' button to activate your webcam  
    2. Perform sign language gestures in view  
    3. The app will recognize, translate, and speak the signs  
    4. Stop the camera when done  
    """)

    if model_loaded and label_encoder is not None:
        st.subheader("Available Signs")
        st.write(", ".join(sorted(label_encoder.classes_)))
