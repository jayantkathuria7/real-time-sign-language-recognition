from typing import Literal
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
from config.paths import BASE_DIR, MODELS_METADATA_FILE, PLOTS_DIR, REPORTS_DIR
from config.model_config import NUM_CLASSES
from src.data.video_resize import count_videos
from src.model.training.data import load_test_data

def get_dir_size(dir_path):
    """Calculate the total size of all files in a directory."""
    return sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())

def format_size(size_bytes):
    """Convert a size in bytes to a human-readable format."""
    for unit in ['Bytes', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0

def get_dataset_metadata(data_path):
    """Collect metadata about the dataset, including size and video count."""
    return {
        "# signs": NUM_CLASSES,
        "total_videos": count_videos(data_path),
        "path": str(data_path.relative_to(BASE_DIR)),
        "size": format_size(get_dir_size(data_path))
        }

def save_confusion_matrix(cm):
    """Save the confusion matrix as a PNG image."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR/ "current_confusion_matrix.png", dpi=300)
    plt.close()

def save_classification_report(report):
    """Save the classification report as a JSON file."""
    save_path = REPORTS_DIR/"classification_report.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Classification report saved to: {save_path}")

def get_evaluation_metrics(y_true, predictions):
    """Calculate classification metrics from true and predicted labels."""
    cm = confusion_matrix(y_true, predictions)
    save_confusion_matrix(cm)
    report = classification_report(y_true, predictions, output_dict=True)
    save_classification_report(report)
    # save confusion matrix as image and classificationreport as json
    return {        
        "test_accuracy": round(report["accuracy"],2),
        "test_macro_precision": round(report["macro avg"]["precision"],2),
        "test_macro_recall": round(report["macro avg"]["recall"],2),
        "test_macro_f1": round(report["macro avg"]["f1-score"],2),
        "test_weighted_precision": round(report["weighted avg"]["precision"],2),
        "test_weighted_recall": round(report["weighted avg"]["recall"],2),
        "test_weighted_f1": round(report["weighted avg"]["f1-score"],2),
        "test_samples": len(y_true)
    }

def get_model_metadata(model, model_name, version_number, training_time, status=Literal["candidate", "production"]):
    """Collect metadata about a trained model and its training details."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_size_bytes = buffer.tell()

    return {
        "timestamp": datetime.now().isoformat(sep=" "), 
        "model_id": model_name.lower().replace(" ","_") + f"_v{version_number}",
        "model_name": model_name,
        "model_size": format_size(model_size_bytes), 
        "version": version_number,
        "status": status,
        "training_time": f"{training_time} seconds"
    }

def save_metrics(new_metrics, save_path=MODELS_METADATA_FILE):
    """Append model metrics to the metadata file."""
    if save_path.exists():
        with open(save_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = []

    metrics.append(new_metrics)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved:",save_path)


def evaluate_model(model):
    """Evaluate a trained model on the test dataset and return its metrics."""
    X_test, y_test = load_test_data()

    probabilities = model.predict(X_test)
    predictions = np.argmax(probabilities, axis=1)
    return get_evaluation_metrics(y_test, predictions)
