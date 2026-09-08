import time
from deep_translator import GoogleTranslator

def translate_words(words, language_name, language_code):
    print(f"Translating words into {language_name.title()}....")

    translator = GoogleTranslator(source="en", target=language_code)
    translations = []

    for word in words:
        for attempt in range(3):
            try:
                translated_text = translator.translate(word)
                print(f"{word!r} -> {translated_text!r}")
                translations.append({"origin": word,"text": translated_text})
                break

            except Exception as e:
                print(f"[{language_name}] "f"{word!r} failed "f"(attempt {attempt + 1}/3)")
                time.sleep(1)

        else:
            print(f"[WARNING] Could not translate " f"{word!r} into {language_name}")

    return translations
