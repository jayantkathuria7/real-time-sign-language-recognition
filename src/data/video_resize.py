import cv2
import os
from tqdm import tqdm
from pathlib import Path
from config.paths import RAW_VIDEO_DIR, RESIZED_VIDEO_DIR
from config.model_config import VIDEO_EXTENSIONS, TARGET_HEIGHT, TARGET_WIDTH

def count_videos(path):
    """Count all video files in a directory and its subdirectories."""
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
    )

def ensure_videos_resized():
    """Resize videos if the resized dataset is missing or invalid."""
    if videos_are_resized():
        print("✓ Videos already resized")
        return
    print("→ Resizing videos...")
    resize_all_videos()

def videos_are_resized():
    """Check whether all videos are resized to the configured dimensions."""
    path = RESIZED_VIDEO_DIR

    if not path.exists() or not path.is_dir():
        return False

    if not any(path.iterdir()):
        return False

    for word_dir in path.iterdir():
        if not word_dir.is_dir():
            continue

        for video_path in word_dir.iterdir():
            if not video_path.is_file():
                continue

            cap = cv2.VideoCapture(str(video_path))

            if not cap.isOpened():
                cap.release()
                return False

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.release()

            if width != TARGET_WIDTH or height != TARGET_HEIGHT:
                return False
    if count_videos(RAW_VIDEO_DIR) != count_videos(RESIZED_VIDEO_DIR):
        return False

    return True

def resize_video(input_path, output_path):
    """Resize a video to the target dimensions and save it to the output path."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        # print(f"Error opening video file: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        out.write(resized_frame)

    cap.release()
    out.release()
    return True

def resize_all_videos():
    """Resize all videos while preserving their directory structure."""
    processed = 0
    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        relative_path = Path(root).relative_to(RAW_VIDEO_DIR)
        output_dir = RESIZED_VIDEO_DIR / relative_path
        output_dir.mkdir(parents=True, exist_ok=True)
        video_files = [file for file in files if Path(file).suffix.lower() in VIDEO_EXTENSIONS]

        for file in tqdm(video_files, desc=f"Processing {relative_path} videos..."):
            input_file_path = Path(root) / file
            output_file_path = output_dir / file
            if resize_video(input_file_path, output_file_path):
                processed += 1
            else:
                print(f"Failed to process: {input_file_path}")

    print(f"Finished. {processed} videos processed.")
