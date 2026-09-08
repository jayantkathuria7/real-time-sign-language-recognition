import os
import pygame
from config.languages import KEYS_CODES

def handle_keyboard_input(key, state, selected_language):
    """Handle keyboard commands for quitting, language selection, and audio playback."""
    should_quit=False
    if key == ord('q'):
        should_quit=True

    elif key in {ord('h'), ord('g'), ord('p'), ord('u')}:
        selected_language = KEYS_CODES.get(chr(key))
        print(f"Switched to {selected_language}")
        # Reset audio control variables on language change
        state.last_played_sign = ""
        state.stable_detection_frames = 0

    elif key == ord('a'):  # Toggle automatic audio playback
        state.auto_play_audio = not state.auto_play_audio
        print(f"Automatic audio playback: {'ON' if state.auto_play_audio else 'OFF'}")
        
    elif key == ord(' '):  # Spacebar to manually play audio for current prediction
        if state.current_prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
            audio_file = f"audio/{selected_language}/{state.current_prediction}.mp3"
            if os.path.exists(audio_file):
                try:
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                    print(f"Manually playing audio for: {state.current_prediction}")
                except Exception as e:
                    print(f"Error playing audio: {e}")
            else:
                print(f"Audio file not found: {audio_file}")

    return should_quit, selected_language
