import numpy as np
from config.paths import ARTIFACTS_DIR
from config.model_config import EPOCHS, BATCH_SIZE
from src.model.training.data import compute_class_weights, prepare_training_data
from src.model.training.callbacks import get_callbacks

def train_model(model):
    """Train the model on prepared data and save the test dataset for evaluation."""
    X_train_new, X_test, y_train_new, y_test, label_encoder = prepare_training_data()
    early_stopping, lr_scheduler, reduce_lr = get_callbacks()
    class_weights = compute_class_weights(y_train_new)
    training_history = model.fit(
        X_train_new, y_train_new,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    np.savez(ARTIFACTS_DIR / "test_data.npz", X_test=X_test, y_test=y_test)

    return training_history, model, label_encoder



