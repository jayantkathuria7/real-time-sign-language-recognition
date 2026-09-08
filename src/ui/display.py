from PIL import Image, ImageDraw, ImageFont
import cv2
import sys
import numpy as np
from config.app_config import REQUIRED_STABLE_FRAMES
from config.paths import FONT_PATHS
from config.languages import LANGUAGE_CODES

def display_frame(frame, language, fps, prediction, auto_play_audio, stable_detection_frames, confidence, last_translated):
    """Display the camera frame with recognition, translation, and audio status information."""
    # Create overlay with PIL for better font rendering
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Load font for translation
    lang_code = LANGUAGE_CODES.get(language)
    font_path = FONT_PATHS.get(lang_code)

    eng_font_path = FONT_PATHS.get("en", 'arial.ttf')
    height, width = frame.shape[:2]
    font_size = int(height * 0.04)
    eng_font_size = int(height * 0.04)

    try:
        font = ImageFont.truetype(font_path, font_size)
        eng_font = ImageFont.truetype(eng_font_path, eng_font_size)
    except IOError as e:
        # Fallback to default font
        font = ImageFont.load_default(font_size)
        eng_font = ImageFont.load_default(eng_font_size)
        raise RuntimeError(f"Could not load font: {font_path}") from e

    # Display FPS
    draw.text((10, 30), f"FPS: {fps:.1f}", font=eng_font, fill=(0, 255, 0))
    
    # Display text with status indicators
    draw.text((10, 90), f"Sign: {prediction}", font=eng_font, fill=(255, 0, 255))
    
    # Add audio status indicator
    audio_mode_status = "Auto" if auto_play_audio else "Manual"
    audio_ready_status = ""
    if stable_detection_frames > 0 and auto_play_audio:
        audio_ready_status = f" (Ready in {REQUIRED_STABLE_FRAMES - stable_detection_frames})" if stable_detection_frames < REQUIRED_STABLE_FRAMES else " (Ready)"
    
    # Display confidence if a sign is detected
    if prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
        draw.text((10, 150), f"Confidence: {confidence:.2f}", font=eng_font, fill=(255, 0, 0))
        draw.text((10, 270), f"Audio: {audio_mode_status}{audio_ready_status}", font=eng_font, fill=(0, 200, 200))
    
    # Display language name
    draw.text((250, 30), f"Language: {language.capitalize()}", font=eng_font, fill=(255, 255, 255))
    
    # Display translation
    if last_translated and prediction not in ["No hand detected", "No sign detected", "Unknown sign"]:
        draw.text((10, 210), last_translated, font=font, fill=(255, 255, 0))
    
    # Convert back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)    
    cv2.imshow('Real-time Sign Detection', frame)