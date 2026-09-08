import os
import pygame
import time
from gtts import gTTS
from config.app_config import REQUIRED_STABLE_FRAMES, AUDIO_COOLDOWN
from config.paths import AUDIO_DATA_DIR

def play_audio(audio_file, current_prediction, current_time):
    """Play a saved audio file and return the sign and time it was played."""
    last_played_sign = ""
    last_played_time = 0
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

    return last_played_sign, last_played_time


def maybe_play_audio(state, language):
    """Play audio automatically when a detected sign is stable and eligible."""
    current_time = time.time()

    if (state.auto_play_audio 
        and state.stable_detection_frames >= REQUIRED_STABLE_FRAMES 
        and (state.current_prediction != state.last_played_sign 
            or current_time - state.last_played_time > AUDIO_COOLDOWN)):
        # Construct the audio file path
        audio_file = AUDIO_DATA_DIR / language / f"{state.current_prediction}.mp3"
        state.last_played_sign, state.last_played_time = play_audio(audio_file, state.current_prediction, current_time)

def speak_text(text, lang_code):
    """Convert text to speech, play it, and remove the temporary audio file."""
    try:
        tts = gTTS(text=text, lang=lang_code)
        filename = "temp_audio.mp3"
        tts.save(filename)
        pygame.mixer.init()
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(filename)
    except Exception as e:
        print(f"[Speech Error]: {e}")
