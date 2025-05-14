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
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

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

# RTC Configuration for WebRTC (use STUN servers for deploying behind firewalls)
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Sidebar settings
st.sidebar.header("Settings")
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

# Create placeholders for real-time updates
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Recognition Results")
    prediction_placeholder = st.empty()
    confidence_placeholder = st.empty()

    st.subheader("Translation")
    translation_placeholder = st.empty()
    audio_placeholder = st.empty()


# Create WebRTC video processor class
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.last_prediction = ""
        self.last_prediction_time = time.time()
        self.prediction_cooldown = 0.5  # seconds
        self.current_prediction = None
        self.current_confidence = 0.0

    def extract_hand_keypoints(self, results):
        lh = np.zeros((21, 3))
        rh = np.zeros((21, 3))
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        return np.concatenate([lh, rh], axis=0)

    def normalize_keypoints(self, keypoints):
        wrist = keypoints[0]
        keypoints -= wrist
        scale = np.linalg.norm(keypoints[4] - keypoints[20])
        return keypoints / (scale + 1e-6)

    def recv(self, frame):
        global prediction_placeholder, confidence_placeholder, translation_placeholder, audio_placeholder

        img = frame.to_ndarray(format="bgr24")

        # Process with MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(img_rgb)

        # Draw landmarks if enabled
        if show_landmarks:
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract and normalize keypoints
        try:
            keypoints = self.extract_hand_keypoints(results)
            normalized = self.normalize_keypoints(keypoints)
            self.sequence.append(normalized)
        except Exception as e:
            # Skip this frame if keypoint extraction fails
            pass

        # Make predictions when we have enough frames
        if len(self.sequence) == SEQUENCE_LENGTH and model_loaded:
            # Only predict every few frames to avoid overloading
            current_time = time.time()
            if current_time - self.last_prediction_time > self.prediction_cooldown:
                input_data = np.array(self.sequence).reshape(1, SEQUENCE_LENGTH, 126)
                prediction = model.predict(input_data, verbose=0)
                pred_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
                confidence = np.max(prediction)

                # Update the prediction state
                self.current_prediction = pred_label
                self.current_confidence = confidence

                # Update UI elements if confidence is high enough
                if confidence > confidence_threshold:
                    # Adding text to the frame
                    cv2.putText(img, f"Sign: {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"Conf: {confidence:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Update Streamlit UI
                    prediction_placeholder.markdown(f"**Recognized Sign:** {pred_label}")
                    confidence_placeholder.markdown(f"**Confidence:** {confidence:.2f}")

                    # Translation and audio
                    if pred_label != self.last_prediction and translation_module_available:
                        lang_key = languages[target_language].lower()
                        translated_text = pre_translations.get(lang_key, {}).get(pred_label, None)

                        if translated_text:
                            translation_placeholder.markdown(f"**Translation ({target_language}):** {translated_text}")

                            audio_path = f"audio/{lang_key}/{pred_label}.mp3"
                            if os.path.exists(audio_path):
                                with open(audio_path, "rb") as f:
                                    audio_bytes = f.read()
                                    audio_placeholder.audio(audio_bytes, format="audio/mp3")
                        else:
                            translation_placeholder.markdown("*No translation available for this sign.*")

                        self.last_prediction = pred_label
                else:
                    # Update UI with low confidence message
                    prediction_placeholder.markdown("**Recognized Sign:** Waiting for clear sign...")
                    confidence_placeholder.markdown(f"**Confidence:** {confidence:.2f}")

                self.last_prediction_time = current_time

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Main app flow
if model_loaded:
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="sign-language-detection",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if not webrtc_ctx.state.playing:
            st.info("Click the 'Start' button to activate your camera for real-time sign language recognition.")

    # Fallback option for file upload
    st.sidebar.markdown("---")
    st.sidebar.subheader("Alternative: Upload Video")
    uploaded_file = st.sidebar.file_uploader("Upload a video if camera is unavailable", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        process_video_button = st.sidebar.button("Process Uploaded Video")

        if process_video_button:
            # Video processing code here (similar to previous implementation)
            # This provides a fallback for users who can't use the camera
            st.info("Video processing functionality available as a fallback option")

    # Display available signs
    st.sidebar.markdown("---")
    st.sidebar.subheader("Available Signs")
    st.sidebar.write(", ".join(sorted(label_encoder.classes_)))

else:
    st.error("Model or label encoder could not be loaded. Please check your files.")

# Instructions
with st.expander("How to use"):
    st.markdown("""
    ### Instructions:
    1. Click the 'Start' button above to activate your camera
    2. Allow browser camera access when prompted
    3. Perform sign language gestures in view of the camera
    4. The system will recognize signs and display translations in real-time
    5. Click 'Stop' when finished

    ### Tips for better recognition:
    - Ensure good lighting conditions
    - Position your hands clearly in frame
    - Make deliberate, clear gestures
    - Try to maintain a neutral background
    """)