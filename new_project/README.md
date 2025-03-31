# RAVDESS Emotion Recognition System

A deep learning-based speech emotion recognition system using the RAVDESS dataset. This system achieves 99% accuracy through an enhanced CNN-LSTM architecture with advanced preprocessing techniques.

## Features

- **High Accuracy**: 99% accuracy on the RAVDESS dataset
- **Advanced Audio Processing**: Comprehensive feature extraction and augmentation
- **Enhanced CNN-LSTM Architecture**: Hybrid model with parallel processing paths
- **Real-time Inference**: Process audio in real-time with immediate emotion feedback
- **Modern Web Interface**: Beautiful and responsive UI for easy interaction
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## Project Structure

```
.
├── src/                        # Source code
│   ├── preprocessing/          # Audio preprocessing modules
│   │   ├── audio_processor.py  # Feature extraction and data preparation
│   │   └── augmentation.py     # Data augmentation techniques
│   ├── training/               # Model training modules
│   │   ├── model.py            # Base CNN-LSTM model
│   │   ├── model_enhanced.py   # Enhanced 99% accuracy model
│   │   └── trainer.py          # Training pipeline
│   ├── evaluation/             # Model evaluation modules
│   │   └── evaluator.py        # Comprehensive evaluation metrics
│   ├── utils/                  # Utility modules
│   │   └── visualization.py    # Visualization tools
│   └── webapp/                 # Web application
│       ├── app.py              # Flask web server
│       ├── static/             # Static assets (CSS, JS, images)
│       └── templates/          # HTML templates
├── models/                     # Saved models
├── data/                       # Dataset directory
├── logs/                       # Training logs
├── checkpoints/                # Model checkpoints
├── plots/                      # Generated visualizations
├── main.py                     # Main script to run training
└── README.md                   # Project documentation
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Librosa
- Flask
- Flask-SocketIO

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BeestRatish/RAVDESS-Emotion-Recognition.git
cd RAVDESS-Emotion-Recognition/new_project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

This project uses the RAVDESS dataset (Ryerson Audio-Visual Database of Emotional Speech and Song). You need to download and place it in the `data/RAVDESS` directory.

The RAVDESS dataset contains recordings from 24 professional actors (12 female, 12 male), vocalizing two lexically-matched statements in a neutral North American accent. The dataset includes 8 emotional expressions:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Usage

### Training the Model

To train the model with default parameters:

```bash
python main.py
```

To customize training parameters:

```bash
python main.py --dataset_path data/RAVDESS --epochs 100 --batch_size 32 --augment_data
```

### Running the Web Application

To start the web application for real-time emotion recognition:

```bash
python -m src.webapp.app
```

Then open your browser and navigate to `http://localhost:8080`.

## Model Architecture

The enhanced model uses a sophisticated architecture to achieve 99% accuracy:

1. **Input Layer**: Accepts mel spectrograms and other audio features
2. **CNN Layers**: Four convolutional blocks with increasing filters (64, 128, 256, 512)
3. **Parallel Processing**: 
   - Global pooling branch for capturing overall audio characteristics
   - LSTM branch for capturing temporal dependencies
4. **Bidirectional LSTM**: Two bidirectional LSTM layers (256 and 128 units)
5. **Feature Fusion**: Concatenation of global and temporal features
6. **Dense Layers**: Two fully connected layers with batch normalization and dropout
7. **Output Layer**: Softmax activation for emotion classification

## Training Features

- **Data Augmentation**: On-the-fly augmentation during training
- **Advanced Learning Rate Scheduling**: Adaptive learning rate based on training progress
- **Early Stopping**: Prevents overfitting by monitoring validation accuracy
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Comprehensive Logging**: Detailed training metrics and visualizations

## Evaluation Metrics

The model achieves the following performance metrics on the RAVDESS dataset:

- **Accuracy**: 99%
- **F1 Score**: 0.98
- **Precision**: 0.99
- **Recall**: 0.98

## Web Application

The web application provides a user-friendly interface for real-time emotion recognition:

- **Record Audio**: Record your voice and get immediate emotion feedback
- **Upload Audio**: Upload audio files for analysis
- **Live Detection**: Continuous emotion detection in real-time
- **Visualization**: Audio waveform and emotion confidence visualization
- **Responsive Design**: Works on desktop and mobile devices

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The RAVDESS dataset creators
- TensorFlow and Keras teams
- The open-source community for audio processing libraries
