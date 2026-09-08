import numpy as np
from src.model.training.model import load_current_model
from src.model.evaluation.metrics import evaluate_model, get_model_metadata, get_dataset_metadata, save_metrics
from config.paths import RAW_VIDEO_DIR
from config.model_config import MODEL_VERSION

def run_evaluation_pipeline(training_time):
    """Evaluate the current model, collect metadata, and save the results."""
    model, _ = load_current_model()
    evaluation_metrics = evaluate_model(model)
    metadata = get_model_metadata(
        model,
        model_name="CNN-LSTM",
        version_number=MODEL_VERSION,
        training_time=training_time,
        status="production"
    )
    metadata["dataset"] = get_dataset_metadata(data_path=RAW_VIDEO_DIR)
    metadata["evaluation_metrics"] = evaluation_metrics
    save_metrics(metadata)