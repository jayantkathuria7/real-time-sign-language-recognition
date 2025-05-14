import os
import cv2
from tqdm import tqdm

input_base_path = 'data/videos'
output_base_path = 'data/resized_videos'

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

target_width = 800
target_height = 600

def resize_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

    for _ in tqdm(range(frame_count), desc=f"Processing {os.path.basename(input_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (target_width, target_height))

        out.write(resized_frame)

    cap.release()
    out.release()
    return True

for root, dirs, files in os.walk(input_base_path):
    relative_path = os.path.relpath(root, input_base_path)
    output_dir = os.path.join(output_base_path, relative_path)
    if not os.path.exists(output_dir) and relative_path != '.':
        os.makedirs(output_dir)

    for file in files:
        if file.endswith('.mp4'):
            input_file_path = os.path.join(root, file)
            output_file_path = os.path.join(output_dir, file)
            print(f"Resizing: {input_file_path}")
            success = resize_video(input_file_path, output_file_path)
            if success:
                print(f"Successfully resized and saved to: {output_file_path}")
            else:
                print(f"Failed to process: {input_file_path}")

print("All videos have been processed!")