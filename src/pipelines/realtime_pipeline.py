import cv2
import mediapipe as mp
import time
import pygame  # For audio playback
from config.app_state import AppState
from config.app_config import CAMERA_INDEX
from config.model_config import TARGET_HEIGHT, TARGET_WIDTH
from src.ui.display import display_frame
from src.ui.landmarks import draw_landmarks
from src.input.keyboard_input import handle_keyboard_input
from src.recognition.prediction import handle_prediction
from src.model.training.model import load_current_model, load_current_encoder
from src.translations.translation_store import load_translations

def initialize_camera():
    """Initialize and return the configured camera."""
    return cv2.VideoCapture(CAMERA_INDEX)

def initialize_mediapipe():
    """Initialize and return the MediaPipe Holistic and drawing utilities."""
    mp_holistic = mp.solutions.holistic
    return mp_holistic

def run_realtime_recognition():
    """Run the real-time sign language recognition pipeline."""
    model, model_id = load_current_model()
    encoder = load_current_encoder(model_id)
    translations, _, error = load_translations()
    if error:
        raise FileNotFoundError(error)

    if model.output_shape[-1] != len(encoder.classes_):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but encoder has {len(encoder.classes_)} classes."
        )

    state = AppState()

    cap = initialize_camera()
    mp_holistic = initialize_mediapipe()
    pygame.mixer.init()
    
    # Initialize language settings
    selected_language = 'hindi'

    with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic_processor:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - state.prev_time) if state.prev_time > 0 else 0
            state.prev_time = current_time

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic_processor.process(frame_rgb)

            frame_with_landmarks, hands_detected = draw_landmarks(frame, results)

            if hands_detected:
                handle_prediction(model, translations, encoder, results, state, selected_language)
            else:
                state.reset_no_hand()

            state.frame_count += 1

            display_frame(frame=frame_with_landmarks, language=selected_language, fps=fps, 
                          prediction=state.current_prediction, auto_play_audio=state.auto_play_audio, 
                          stable_detection_frames=state.stable_detection_frames, 
                          confidence=state.confidence, last_translated=state.last_translated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            shouldquit, selected_language = handle_keyboard_input(key, state, selected_language)
            if shouldquit:
                break

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()  # Clean up pygame resources

if __name__ == "__main__":
    run_realtime_recognition()