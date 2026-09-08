from config.paths import RAW_VIDEO_DIR, AUDIO_DATA_DIR
from config.languages import LANGUAGE_CODES
from src.translations.translator import translate_words
from src.translations.translation_store import save_translations
from src.audio.generator import generate_audio

def create_translations_and_audio():
    """Generate translations and corresponding audio files for all supported languages."""
    words = [file.stem for file in RAW_VIDEO_DIR.iterdir() 
             if file.is_dir() and not file.name.startswith(".")]
    print(words)
    translations = {}

    for language_name, language_code in LANGUAGE_CODES.items():
        language_translations = translate_words(words, language_name, language_code)
        translations[language_name] = {}

        language_audio_path = AUDIO_DATA_DIR / language_name
        language_audio_path.mkdir(parents=True, exist_ok=True)

        for translation in language_translations:
            curr_word = translation["origin"]
            translated_word = translation["text"]
            translations[language_name][curr_word] = translated_word
            audio_path = language_audio_path / f"{curr_word}.mp3"
            generate_audio(translated_word, audio_path, language_code)

    return translations


def generate_translations():
    """Generate translations, create audio files, and save the translation data."""
    translations = create_translations_and_audio()
    save_translations(translations)