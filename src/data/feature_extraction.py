import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def extract_features_3d(sequence):
    """Extract features from 3D sequence data."""
    # Reshape to 3D structure
    sequence_3d = sequence.reshape(sequence.shape[0], -1, 3)
    num_keypoints = sequence_3d.shape[1]  # Should be 48 keypoints
    
    t = np.arange(sequence_3d.shape[0]).reshape(-1, 1)
    poly = PolynomialFeatures(degree=2)
    features = []

    for kp_idx in range(num_keypoints):
        for dim in range(3):  # x, y, z
            series = sequence_3d[:, kp_idx, dim]

            # Polynomial regression
            X_poly = poly.fit_transform(t)
            reg = LinearRegression().fit(X_poly, series)
            features.extend(reg.coef_[1:])

            # Velocity & Acceleration
            vel = np.diff(series)
            acc = np.diff(vel) if len(vel) > 1 else np.array([0])
            features.extend([
                np.mean(vel), np.std(vel),
                np.mean(acc), np.std(acc)
            ])

            # FFT
            fft = np.abs(np.fft.fft(series))
            features.extend(fft[:3])

    return np.array(features)

def extract_features_batch(sequences):
    """Extract features from a batch of sequences."""
    print("Extracting features from sequences...")
    return np.array([extract_features_3d(seq) for seq in sequences])
