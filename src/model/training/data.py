import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from config.paths import PROCESSED_DATA_DIR, ARTIFACTS_DIR
import src.data.augmentation as aug

def load_data():
    """Load processed sequences and labels from disk."""
    try:
        X_sequences = np.load(PROCESSED_DATA_DIR / "X_sequences.npy")
        y_labels = np.load(PROCESSED_DATA_DIR / "y_labels.npy")
        return X_sequences, y_labels
    except Exception:
        raise ValueError("The requested record could not be found.")

def load_test_data():
    """Load the saved test dataset from disk."""
    data = np.load(ARTIFACTS_DIR / "test_data.npz")
    X_test = data["X_test"]
    y_test = data["y_test"]
    return X_test, y_test 

def split_data(X, y, test_size=0.2):
    """Split data into stratified training and test sets."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42) 

def encode_target_feature(y):
    """Encode string class labels as integer values."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

def prepare_training_data(augment=True):
    """Load, normalize, split, and optionally augment the training data."""
    X, y = load_data()

    X_reshaped = X.reshape((X.shape[0], X.shape[1], -1))
    X_normalized = np.zeros_like(X_reshaped)
    y_encoded, label_encoder = encode_target_feature(y)

    for i in range(X_reshaped.shape[0]):
        # Normalize each sample to have zero mean and unit variance
        X_normalized[i] = (X_reshaped[i] - np.mean(X_reshaped[i])) / (np.std(X_reshaped[i]) + 1e-8)

    X_train, X_test, y_train, y_test = split_data(X_normalized,y_encoded)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
    if augment==True:
        X_train_new, y_train_new = aug.augment_sequential_data(X_train, y_train)
    else:
        X_train_new, y_train_new = X_train, y_train
    return X_train_new, X_test, y_train_new, y_test, label_encoder

def compute_class_weights(y_train):
    """Calculate class weights to compensate for class imbalance."""
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {i: weight for i, weight in enumerate(class_weights)}