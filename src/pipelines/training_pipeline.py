import time
from src.model.architecture import build_model
from src.model.training.trainer import train_model
from src.model.training.model import save_model
from src.pipelines.evaluation_pipeline import run_evaluation_pipeline
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

def build_and_train_model():
    """Build, train, evaluate, and save the sign language recognition model."""
    model = build_model()
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    start_time = time.perf_counter()
    training_history, trained_model, label_encoder = train_model(model)
    end_time = time.perf_counter()
    training_time = end_time-start_time
    save_model(trained_model, "CNN-LSTM", 1.1, label_encoder)
    run_evaluation_pipeline(training_time)
    print("Model trained and saved.")
