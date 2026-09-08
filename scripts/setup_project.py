from config.paths import RAW_VIDEO_DIR
from src.data.video_resize import ensure_videos_resized
from src.data.dataset_builder import ensure_training_data_exist
from src.model.training.model import model_exists
from src.pipelines.training_pipeline import build_and_train_model
from src.translations.translation_store import translations_exist
from src.pipelines.translation_pipeline import generate_translations

def raw_videos_exist():
    return RAW_VIDEO_DIR.is_dir() and any(RAW_VIDEO_DIR.iterdir())

def ensure_model_exist():
    if model_exists():
        print("✓ Model already exists")
        return
    print("→ Training Model...")
    build_and_train_model()

def ensure_translations_exist():
    if translations_exist():
        print("✓ Translations already exists")
        return
    print("→ Generating Translations...")
    generate_translations()

def main():
    if not raw_videos_exist():
        print("No input videos found")
        return
    print("✓ Input Videos found")
    ensure_videos_resized()
    ensure_training_data_exist()
    ensure_model_exist()
    ensure_translations_exist()

if __name__ =="__main__":
    main()
