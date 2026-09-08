import gtts

def generate_audio(translation, audio_path, language_code):
    """Generate speech from a translated word and save it as an audio file."""
    tts = gtts.gTTS(translation, lang=language_code)
    tts.save(audio_path)
    # print(f"  Saved: {audio_path}")g