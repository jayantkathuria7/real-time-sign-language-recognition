import numpy as np
from tqdm import tqdm
from src.data.keypoints_extraction import read_video
from config.paths import RAW_VIDEO_DIR,RESIZED_VIDEO_DIR, PROCESSED_DATA_DIR
from config.model_config import SEQUENCE_LENGTH
from src.data.video_resize import count_videos

def training_data_exist() -> bool:
    """Check whether valid processed training data already exists."""
    path = PROCESSED_DATA_DIR

    if not path.is_dir() or not any(path.iterdir()):
        return False
    x_path = path / "X_sequences.npy"
    y_path = path / "y_labels.npy"
    if not x_path.is_file() or not y_path.is_file(): 
        return False
    try:
        X = np.load(x_path)
        y = np.load(y_path)
    except Exception:
        return False
    expected_count = count_videos(RAW_VIDEO_DIR)
    if not (X.shape[0] == y.shape[0] == expected_count):
        print(X.shape[0], y.shape[0], expected_count)
        return False
    return True
    
def ensure_training_data_exist():
    """Generate and save training data if it does not already exist."""
    if training_data_exist():
        print("✓ Training Data already exists")
        return
    print("→ Generating Sequences...")
    build_and_save_data()


def process_videos(data_dir=RESIZED_VIDEO_DIR):
    """Extract, pad, and standardize video sequences for model training."""
    print("Processing all videos...")

    sequences, labels = [], []

    # Collect all video files first
    video_files = [
        (word_dir.name, video_file)
        for word_dir in data_dir.iterdir()
        if word_dir.is_dir()
        for video_file in word_dir.iterdir()
        if video_file.is_file()
    ]

    for word, video_file in tqdm(
        video_files,
        desc="Processing videos",
        unit="video"
    ):
        sequence = read_video(video_file)
        seq_len = len(sequence)

        if seq_len < SEQUENCE_LENGTH:
            pad = [np.zeros((42, 3))] * (SEQUENCE_LENGTH - seq_len)
            sequence = sequence + pad

        else:
            middle_index = seq_len // 2
            start_index = max(0, middle_index - SEQUENCE_LENGTH // 2)
            end_index = min(start_index + SEQUENCE_LENGTH, seq_len)

            middle_frames = sequence[start_index:end_index]

            if len(middle_frames) < SEQUENCE_LENGTH:
                pad = [np.zeros((42, 3))] * (SEQUENCE_LENGTH - len(middle_frames))
                middle_frames += pad

            sequence = middle_frames

        sequences.append(np.array(sequence))
        labels.append(word)

    return np.array(sequences), np.array(labels)


def save_data(X, y, save_dir=PROCESSED_DATA_DIR):
    """Save processed sequences and labels as NumPy files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "X_sequences.npy", X)
    np.save(save_dir / "y_labels.npy", y)

def build_and_save_data():
    """Build the training dataset from videos and save it to disk."""
    X, y = process_videos()
    save_data(X, y)
    return X.shape, y.shape