real-time-sign-language-recognition
├── artifacts
│   ├── encoders
│   │   ├── cnn-lstm_v1.0_encoder.joblib
│   │   └── cnn-lstm_v1.1_encoder.joblib
│   ├── plots
│   │   ├── class_distribution.png
│   │   ├── class_distribution_after_augmentation.png
│   │   ├── cnn_lstm_aug_training_history.png
│   │   ├── cnn_lstm_confusion_matrix.png
│   │   ├── cnn_lstm_training_curves_with_tithout_augmentation.png
│   │   ├── cnn_lstm_training_history.png
│   │   ├── confusion_matrix_Logistic Regression.png
│   │   ├── confusion_matrix_Random Forest.png
│   │   ├── confusion_matrix_SVM.png
│   │   ├── current_confusion_matrix.png
│   │   ├── lstm_aug_training_history.png
│   │   ├── lstm_confusion_matrix.png
│   │   ├── lstm_training_curves_with_tithout_augmentation.png
│   │   ├── lstm_training_history.png
│   │   ├── model_accuracies_comparison.png
│   │   ├── model_f1_scores_comparison.png
│   │   ├── model_training_time_comparison.png
│   │   ├── neural_network_confusion_matrix.png
│   │   └── neural_networks_training_history.png
│   ├── reports
│   │   └── classification_report.json
│   └── test_data.npz
├── assets
│   ├── Major Project PPT.pptx
│   └── demo_video.mp4
├── config
│   ├── __init__.py
│   ├── app_config.py
│   ├── app_state.py
│   ├── languages.py
│   ├── model_config.py
│   └── paths.py
├── models
│   ├── archive
│   │   ├── cnn-lstm_v1.0.keras
│   │   └── cnn-lstm_v1.1.keras
│   ├── current
│   │   └── cnn-lstm_v1.1.keras
│   └── models_metadata.json
├── notebook
│   └── model_training.ipynb
├── scripts
│   ├── build_dataset.py
│   ├── build_translations.py
│   ├── resize_videos.py
│   ├── setup_project.py
│   └── train_model.py
├── src
│   ├── audio
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── player.py
│   ├── data
│   │   ├── __init__.py
│   │   ├── augmentation.py
│   │   ├── dataset_builder.py
│   │   ├── feature_extraction.py
│   │   ├── keypoints_extraction.py
│   │   └── video_resize.py
│   ├── input
│   │   └── keyboard_input.py
│   ├── model
│   │   ├── evaluation
│   │   │   └── metrics.py
│   │   ├── training
│   │   │   ├── __init__.py
│   │   │   ├── callbacks.py
│   │   │   ├── data.py
│   │   │   ├── model.py
│   │   │   └── trainer.py
│   │   └── architecture.py
│   ├── pipelines
│   │   ├── __init__.py
│   │   ├── evaluation_pipeline.py
│   │   ├── real_time_recognition.py
│   │   ├── training_pipeline.py
│   │   └── translation_pipeline.py
│   ├── recognition
│   │   └── prediction.py
│   ├── translations
│   │   ├── __init__.py
│   │   ├── translation_store.py
│   │   └── translator.py
│   ├── ui
│   │   ├── display.py
│   │   └── landmarks.py
│   └── __init__.py
├── .gitignore
├── README.md
├── architecture.md
├── file_index.md
├── file_tree.md
└── requirements.txt