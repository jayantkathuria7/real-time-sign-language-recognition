import numpy as np
from scipy.interpolate import interp1d
from config.model_config import SEQUENCE_LENGTH

# --- AUGMENTATION FUNCTIONS ---
def jitter_sequence(seq, sigma=0.01):
    """Add Gaussian noise to a sequence to simulate small variations."""
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def warp_time(sequence, target_len=30):
    """Resample a sequence to a target length using linear interpolation."""
    original_len = sequence.shape[0]
    
    if original_len == target_len:
        return sequence.copy()
    
    t_orig = np.linspace(0, 1, original_len)
    t_new = np.linspace(0, 1, target_len)
    
    warped_sequence = np.zeros((target_len, sequence.shape[1], sequence.shape[2]))
    
    for kp in range(sequence.shape[1]):
        for dim in range(sequence.shape[2]):
            f = interp1d(t_orig, sequence[:, kp, dim], kind='linear')
            warped_sequence[:, kp, dim] = f(t_new)
    
    return warped_sequence

def rotate_hand(sequence, angle_degrees=10, axis='z'):
    """Rotate hand landmark coordinates around the specified axis."""
    angle = np.radians(angle_degrees)
    if axis == 'z':
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0,              0,             1]])
    elif axis == 'y':
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0,             1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(angle), -np.sin(angle)],
                      [0, np.sin(angle),  np.cos(angle)]])

    return np.dot(sequence, R)


# --- SYNTHETIC DATA GENERATION ---
def generate_augmented_dataset(X_raw, y):
    """Generate an augmented dataset using noise, time warping, and rotation."""
    print('Generating Augmented data.....')
    X_aug = []
    y_aug = []
    for i, seq in enumerate(X_raw):
        X_aug.append(seq)
        y_aug.append(y[i])

        X_aug.append(jitter_sequence(seq))
        y_aug.append(y[i])

        X_aug.append(warp_time(seq, SEQUENCE_LENGTH))
        y_aug.append(y[i])

        X_aug.append(rotate_hand(seq, angle_degrees=10, axis='z'))
        y_aug.append(y[i])

    return np.array(X_aug), np.array(y_aug)

def augment_sequential_data(X, y):
    """Augment sequential data using noise, temporal shifting, and scaling."""
    print('Augmenting LSTM-compatible data...')
    augmented_X = []
    augmented_y = []

    augmented_X.append(X)
    augmented_y.append(y)

    # Noise
    noise = np.random.normal(0, 0.05, X.shape)
    augmented_X.append(X + noise)
    augmented_y.append(y)

    # Time warp (simple shift)
    warped_X = np.zeros_like(X)
    for i in range(X.shape[0]):
        shift = np.random.randint(1, 3)
        for j in range(X.shape[1]):
            if j < X.shape[1] - shift:
                warped_X[i, j] = X[i, j + shift]
            else:
                warped_X[i, j] = X[i, j]
    augmented_X.append(warped_X)
    augmented_y.append(y)

    # Scaling
    scale_factors = np.random.uniform(0.8, 1.2, (X.shape[0], 1, 1))
    scaled_X = X * scale_factors
    augmented_X.append(scaled_X)
    augmented_y.append(y)
    final_X = np.vstack(augmented_X)
    final_y = np.concatenate(augmented_y)

    return final_X, final_y
