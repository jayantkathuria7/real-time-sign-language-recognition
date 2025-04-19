import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator
from gtts import gTTS
import pygame

language_codes = {
    'hindi': 'hi',
    'punjabi': 'pa',
    'urdu': 'ur',
    'gujarati': 'gu'
}

font_paths = {
    'hi': 'fonts/NotoSansDevanagari-Regular.ttf',
    'pa': 'fonts/NotoSansGurmukhi-Regular.ttf',
    'ur': 'fonts/NotoSansArabic-Regular.ttf',
    'gu': 'fonts/NotoSansGujarati-Regular.ttf'
}

def translate_text(text, target_language='hindi'):
    try:
        lang_code = language_codes.get(target_language.lower())
        if not lang_code:
            raise ValueError(f"Unsupported language: {target_language}")
        translator = Translator()
        translated = translator.translate(text, src='en', dest=lang_code).text
        return translated, lang_code
    except Exception as e:
        print(f"[Translation Error]: {e}")
        return None, None

def generate_text_image(text, lang_code, width=800, height=600, font_size=50):
    try:
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        font_path = font_paths[lang_code]
        font = ImageFont.truetype(font_path, font_size)
        draw.text((10, 100), text, font=font, fill=(0, 0, 0))
        return np.array(pil_image)
    except Exception as e:
        print(f"[Image Error]: {str(e).encode(sys.stdout.encoding, errors='replace').decode()}")
        return None

def speak_text(text, lang_code):
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
