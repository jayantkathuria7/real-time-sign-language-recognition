from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_VIDEO_DIR = DATA_DIR / "videos"

RESIZED_VIDEO_DIR = DATA_DIR / "resized_videos"
PROCESSED_DATA_DIR = DATA_DIR / "sequences"

TRANSLATIONS_DATA_DIR = DATA_DIR / "translations"
AUDIO_DATA_DIR = DATA_DIR / "audio"

FONT_DIR = DATA_DIR / "fonts"
FONT_PATHS = {
    "en": FONT_DIR / "NotoSans-Regular.ttf",
    'hi': FONT_DIR / "NotoSansDevanagari-Regular.ttf",
    'gu': FONT_DIR / "NotoSansGujarati-Regular.ttf",
    'pa': FONT_DIR / "NotoSansGurmukhi-Regular.ttf",
    'ur': FONT_DIR / "NotoNastaliqUrdu-Regular.ttf"
}

MODELS_DIR = BASE_DIR / "models" 
CURRENT_MODEL_DIR = MODELS_DIR / "current"
ARCHIVED_MODELS_DIR = MODELS_DIR / "archive"
MODELS_METADATA_FILE = MODELS_DIR / "models_metadata.json"

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

directories = [DATA_DIR, RAW_VIDEO_DIR, RESIZED_VIDEO_DIR, PROCESSED_DATA_DIR, 
               TRANSLATIONS_DATA_DIR, AUDIO_DATA_DIR, FONT_DIR,
               MODELS_DIR, CURRENT_MODEL_DIR, ARCHIVED_MODELS_DIR, 
               ARTIFACTS_DIR, ENCODERS_DIR, PLOTS_DIR, REPORTS_DIR, ]

for directory in directories:
    directory.mkdir(parents=True, exist_ok=True)
