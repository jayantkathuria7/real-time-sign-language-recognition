import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.layers import Masking, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def build_mlp_classifier(input_shape, num_classes):
    """Build a simple MLP classifier."""
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm_model(sequence_length, feature_dim, num_classes):
    """Build a bidirectional LSTM model."""
    model = Sequential([
        # Masking layer for any padded sequences
        Masking(mask_value=0., input_shape=(sequence_length, feature_dim)),
        
        # First bidirectional LSTM layer
        Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second bidirectional LSTM layer
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third bidirectional LSTM layer
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        
        # Fully connected layers
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_cnn_lstm_model(sequence_length, feature_dim, num_classes):
    """Build a CNN-LSTM hybrid model."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, feature_dim)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        
        Bidirectional(LSTM(64)),
        BatchNormalization(),
        
        Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_callbacks():
    """Get callbacks for model training."""
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    def cosine_decay_with_warmup(epoch, total_epochs=100, warmup_epochs=5):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs * 0.001
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.0001 + 0.0009 * 0.5 * (1 + np.cos(np.pi * progress))
    
    lr_scheduler = LearningRateScheduler(cosine_decay_with_warmup)
    
    return [early_stopping, lr_scheduler, reduce_lr]

def compute_class_weights(y_train):
    """Compute class weights for imbalanced datasets."""
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    return {i: weight for i, weight in enumerate(class_weights)}

def get_classical_models():
    """Get classical ML models."""
    models = {
        'logistic_regression': LogisticRegression(max_iter=2000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True)
    }
    return models

def ensemble_predict(models, X):
    """Create ensemble prediction from multiple models."""
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    return np.argmax(ensemble_pred, axis=1)