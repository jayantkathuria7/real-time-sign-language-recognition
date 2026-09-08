from dataclasses import dataclass, field
from src.translations.translation_store import get_translation

@dataclass
class AppState:
    prev_time: int = 0
    predictions_buffer: list = field(default_factory=list)
    frame_count: int = 0

    current_prediction: str = "No hand detected"
    confidence: float = 0.0
    last_translated: str = ""
    
    # Audio playback control variables
    last_played_sign: str = ""
    last_played_time: float = 0
    stable_detection_frames: int = 0
    auto_play_audio: bool = True  # Flag to toggle automatic audio playback

    def reset_no_hand(self):
        self.current_prediction = "No hand detected"
        self.confidence = 0.0
        self.last_translated = ""
        self.stable_detection_frames = 0
    
    def update_prediction(self, translations, prediction, avg_confidence, language):
        if prediction is None:
            self.reset_no_hand()
            self.confidence = avg_confidence
            return 
        if prediction == self.current_prediction:
            self.stable_detection_frames += 1
        else:
            self.stable_detection_frames = 0
        
        self.current_prediction = prediction
        self.confidence = avg_confidence
        self.last_translated = get_translation(translations, self.current_prediction, language)
