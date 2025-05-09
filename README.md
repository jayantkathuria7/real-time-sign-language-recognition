# Sign Language Recognition System – Project Workflow

This project is a complete pipeline for recognizing sign language gestures in real time using deep learning, computer vision, and text-to-speech technologies. The central orchestrator is `main.py`, which integrates all components from data collection to gesture recognition and voice output.

---

## 1. Data Collection & Preprocessing

### extracting_and_storing_kkeypoints.py
- Utilizes the **MediaPipe Holistic** model to extract keypoints from body, face, and hands.
- Captures video input frame-by-frame via pre-recorded videos.
- Keypoints are stored in `.npy` (NumPy array) format for efficient storage and training.
- Organizes data into labeled directories based on gesture class for supervised learning.

### resize.py
- Ensures consistency in input data by resizing all video frames to a fixed resolution.
- Preprocessing helps standardize model inputs and improves keypoint extraction accuracy.
- Can be applied as a batch processor for dataset normalization.

---

## 2. Data Augmentation

### augmentation.py
- Expands the dataset by creating new variations of existing sequences.
- Techniques include:
  - Time warping (speeding up or slowing down sequences)
  - Noise injection (simulating minor motion jitters)
  - Cropping or rotating keypoints
- Helps prevent overfitting and improves the model’s ability to generalize to unseen gestures.

---

## 3. Feature Engineering

### feature_extraction.py
- Transforms raw keypoints into meaningful **feature vectors**.
- Includes calculations such as:
  - Normalization of coordinates
  - Angles between joints
  - Distances between key landmarks
- Output is structured to feed directly into the LSTM model for training.

---


## 4. Real-Time Recognition

### real_time_recognition.py
- Uses a webcam to continuously capture live video.
- Processes each frame through the MediaPipe pipeline to extract keypoints.
- Feeds live keypoints into the trained CCNN-LSTM model for inference.
- Predicts the gesture class and displays it in the interface.

---

## 5. Translation & Speech

### pregenerat_translate.py
- Maps gesture labels (like "hello", "thank you") to human-readable strings.
- Supports easy modification or expansion of gesture-to-text mappings.

### translate_and_speak.py
- Uses a **text-to-speech (TTS)** engine (e.g., pyttsx3, gTTS) to convert recognized text into spoken language.
- Provides real-time auditory feedback for accessibility and communication enhancement.
- Can be configured for different languages and voices.

---

## 6. Model Training

### main.py
- The central driver for training the gesture recognition model.
- Responsibilities:
  - Loads preprocessed .npy keypoint data.
  - Splits data into training and validation sets.
  - Loads or builds the LSTM and CNN-LSTM model.
  - Trains the model using defined hyperparameters.
  - Evaluates model performance and saves the trained model (.keras).
- Includes training callbacks (e.g., EarlyStopping, ModelCheckpoint).

---

## Summary Pipeline
Video Input → Frame Resize → Keypoint Extraction → Feature Vector → CNN-LSTM Model → Gesture Prediction → Text Translation → Speech Output

---

## Technologies Used

- **Python**
- **MediaPipe**
- **TensorFlow / Keras**
- **NumPy, OpenCV**
- **Text-to-Speech (pyttsx3 / gTTS)**

---
