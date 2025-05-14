import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.utils.class_weight import compute_class_weight



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