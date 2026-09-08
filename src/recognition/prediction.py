import numpy as np
from collections import deque
from src.data.keypoints_extraction import extract_hand_keypoints, normalize_keypoints
from config.model_config import SEQUENCE_LENGTH, NUM_KEYPOINTS
from config.app_config import PRED_EVERY_N_FRAMES, SMOOTHING_WINDOW
from src.audio.player import maybe_play_audio

# Setup frame buffer
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

def preprocess_for_model(sequence):
    """Convert a keypoint sequence into the shape expected by the model."""
    sequence = np.array(sequence)
    sequence = sequence.reshape(1, SEQUENCE_LENGTH, NUM_KEYPOINTS * 3)
    return sequence

def get_smoothed_prediction(frame_buffer, model, encoder, predictions_buffer):
    """Predict a sign from buffered frames and smooth predictions over time."""
    input_sequence = preprocess_for_model(list(frame_buffer))

    probabilities = model.predict(input_sequence, verbose=0)[0]
    predicted_class = np.argmax(probabilities)
    current_confidence = probabilities[predicted_class]

    predictions_buffer.append((predicted_class, current_confidence))
    if len(predictions_buffer) > SMOOTHING_WINDOW:
        predictions_buffer.pop(0)

    class_counts = {}
    confidence_sums = {}
    for cls, conf in predictions_buffer:
        class_counts[cls] = class_counts.get(cls, 0) + 1
        confidence_sums[cls] = confidence_sums.get(cls, 0) + conf

    smooth_class = max(class_counts, key=class_counts.get)
    avg_confidence = confidence_sums[smooth_class] / class_counts[smooth_class]
    if avg_confidence <= 0.5:
        return None, avg_confidence
    predicted_word = encoder.inverse_transform([int(smooth_class)])[0]
    return predicted_word, avg_confidence
    
def handle_prediction(model, translations, encoder, results, state, language):
    """Extract keypoints, update the frame buffer, and process a new prediction."""
    hand_keypoints = extract_hand_keypoints(results)
    normalized = normalize_keypoints(hand_keypoints)
    frame_buffer.append(normalized)

    if len(frame_buffer) == SEQUENCE_LENGTH and state.frame_count % PRED_EVERY_N_FRAMES == 0:
        new_prediction, avg_confidence = get_smoothed_prediction(frame_buffer, model, encoder, state.predictions_buffer)
        state.update_prediction(translations, new_prediction, avg_confidence, language)
        # Play audio if the sign has been stable and enough time has passed
        maybe_play_audio(state, language)
