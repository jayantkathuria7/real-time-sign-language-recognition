import json
from config.paths import AUDIO_DATA_DIR, TRANSLATIONS_DATA_DIR
from config.languages import SUPPORTED_LANGUAGES
            
def save_translations(translations, save_dir=TRANSLATIONS_DATA_DIR):
    """Save translations to a JSON file."""
    # Save translations to a JSON file
    with open(save_dir / "translations.json", "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)

def get_translation(translations, word, language='hindi'):
    """Return the translation of a word in the selected language."""
    if language in translations and word in translations[language]:
        return translations[language][word]
    return word

def load_translations():
    """Load saved translations and return their availability status."""
    try:
        with open(TRANSLATIONS_DATA_DIR / "translations.json", "r", encoding="utf-8") as f:
            pre_translations = json.load(f)
        translation_module_available = True
        return pre_translations, True, None
    except Exception as e:
        pre_translations = {}
        translation_module_available = False
        return pre_translations, translation_module_available, str(e)

def language_translations_exist():
    """Check whether valid translation data exists for all supported languages."""
    path = TRANSLATIONS_DATA_DIR
    if not path.is_dir() or not any(path.iterdir()):
        return False
    for item in path.iterdir():
        if not item.is_file() or item.suffix.lower() != ".json":
            return False
    try:
        with open(path/"translations.json", "r") as file:
            translations = json.load(file)
            if set(translations.keys()) != SUPPORTED_LANGUAGES:
                return False
    except:
        return False
    return True 

def audio_translations_exist():
    """Check whether audio files exist for all supported languages."""
    path = AUDIO_DATA_DIR
    languages_found = []
    if not path.is_dir() or not any(path.iterdir()):
        return False
    for item in path.iterdir():
        if not item.is_dir():
            return False
        languages_found.append(item.name)
        for file in item.iterdir():
            if not file.is_file() or file.suffix.lower() != ".mp3":
                return False
    if set(languages_found) != SUPPORTED_LANGUAGES:
        return False
    return True 

def translations_exist():
    """Check whether both translation data and audio files are available."""
    return language_translations_exist() and audio_translations_exist()
