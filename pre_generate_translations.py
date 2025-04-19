import os
import json
import gtts
from googletrans import Translator

# Create directories for storing translations and audio
os.makedirs("translations", exist_ok=True)
os.makedirs("audio", exist_ok=True)

# List of sign language words to translate
words = os.listdir('data/videos')

# Target languages
languages = {
    "hindi": "hi",
    "gujarati": "gu",
    "punjabi": "pa",
    "urdu": "ur"
}

# Initialize translator
translator = Translator()

# Store translations
translations = {}

# Generate translations and speech files
for language_name, language_code in languages.items():
    translations[language_name] = {}
    
    # Create language-specific audio directory
    lang_audio_dir = os.path.join("audio", language_name)
    os.makedirs(lang_audio_dir, exist_ok=True)
    
    for word in words:
        print(f"Processing '{word}' in {language_name}...")
        
        # Translate the word
        translation = translator.translate(word, dest=language_code).text
        translations[language_name][word] = translation
        
        # Generate and save speech
        tts = gtts.gTTS(translation, lang=language_code)
        audio_path = os.path.join(lang_audio_dir, f"{word}.mp3")
        tts.save(audio_path)
        print(f"  Saved: {audio_path}")

# Save translations to a JSON file
with open("translations/translations.json", "w", encoding="utf-8") as f:
    json.dump(translations, f, ensure_ascii=False, indent=2)

print("All translations and audio files have been generated!")

