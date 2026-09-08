import shutil
import joblib
from config.paths import CURRENT_MODEL_DIR, ARCHIVED_MODELS_DIR, ARTIFACTS_DIR, ENCODERS_DIR
from tensorflow.keras.models import load_model

def load_current_model():
    """Load the current Keras model and return it with its model identifier."""
    model_files = list(CURRENT_MODEL_DIR.glob("*.keras"))
    if not model_files:
        raise FileNotFoundError(f"No .keras model files found in {CURRENT_MODEL_DIR}")
    
    latest_model_path = model_files[0] 
    try:
        return load_model(latest_model_path), latest_model_path.stem
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {latest_model_path}. Error: {e}")

def model_exists():
    """Check whether a valid Keras model exists in the current model directory."""
    path = CURRENT_MODEL_DIR
    if not path.is_dir() or not any(path.iterdir()):
        return False
    for item in path.iterdir():
        if not item.is_file() or item.suffix.lower() != ".keras":
            return False
    return True

def archive_models():
    """Move the current model to the model archive directory."""
    model_files = list(CURRENT_MODEL_DIR.glob("*.keras"))
    if not model_files:
        print("No current model found to archive.")
        return

    current_model = model_files[0]
    shutil.move(current_model, ARCHIVED_MODELS_DIR / current_model.name)

def save_model(model, model_name, version, encoder):
    """Archive the current model and save the new model and label encoder."""
    archive_models()
    model_id = model_name.lower().replace(" ","_") + f"_v{version}"
    joblib.dump(encoder, ENCODERS_DIR / f"{model_id}_encoder.joblib")
    model.save(CURRENT_MODEL_DIR / f"{model_id}.keras")

def load_current_encoder(model_id):
    """Load the label encoder associated with a model."""
    return joblib.load(ENCODERS_DIR / f"{model_id}_encoder.joblib")