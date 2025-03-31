# Speech Emotion Recognition System using RAVDESS Dataset

This project implements a deep learning-based speech emotion recognition system using the RAVDESS dataset. The system uses an enhanced CNN-LSTM architecture to classify different emotions from speech audio, aiming to achieve 99% accuracy.

## Features

- Audio preprocessing and feature extraction
- Data augmentation for underrepresented emotions
- CNN-LSTM model architecture with batch normalization and dropout
- Model checkpointing and training resumption
- Visualization of training metrics and model performance
- TensorFlow Lite conversion for edge device deployment
- Comprehensive evaluation metrics

## Project Structure

The project has been restructured for better organization and maintainability:

```
RAVDESS-Emotion-Recognition/
├── LICENSE
├── README.md
├── RAVDESS/
│   └── dataset/       # Place the RAVDESS dataset here
└── new_project/       # Main project directory
    ├── README.md
    ├── main.py        # Main script for training and evaluation
    ├── run.py         # CLI tool for running different components
    ├── run.sh         # Shell script for easy execution
    ├── requirements.txt
    ├── checkpoints/   # Model checkpoints during training
    ├── data/          # Processed data
    ├── logs/          # Training logs
    ├── models/        # Saved models
    └── src/           # Source code
        ├── preprocessing/
        │   ├── audio_processor.py
        │   └── augmentation.py
        ├── training/
        │   ├── model.py
        │   ├── model_enhanced.py
        │   └── trainer.py
        ├── evaluation/
        │   └── evaluator.py
        ├── utils/
        │   └── visualization.py
        └── webapp/
            ├── app.py
            ├── __main__.py
            ├── static/
            │   ├── css/
            │   ├── js/
            │   └── img/
            └── templates/
```

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Librosa
- WeasyPrint (for PDF generation)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAVDESS-Emotion-Recognition.git
cd RAVDESS-Emotion-Recognition
```

2. Navigate to the new project directory:
```bash
cd new_project
```

3. Make the run script executable:
```bash
chmod +x run.sh
```

## Dataset

The project uses the RAVDESS dataset. You need to download and place it in the `RAVDESS/dataset` directory.

## Step-by-Step Usage Guide

### Method 1: Using the Shell Script (Recommended)

The shell script handles environment setup, dependency installation, and provides an easy interface to run different project components.

1. Train the model:
```bash
./run.sh train --dataset_path ../RAVDESS/dataset --epochs 100 --batch_size 32
```

2. Run the web application:
```bash
./run.sh webapp --port 8080
```

3. Evaluate the trained model:
```bash
./run.sh evaluate
```

### Method 2: Using Python Directly

If you prefer to use Python directly, you can use the run.py script:

1. Set up a Python virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python run.py train --dataset_path ../RAVDESS/dataset --epochs 100 --batch_size 32
```

4. Run the web application:
```bash
python run.py webapp --port 8080
```

5. Evaluate the trained model:
```bash
python run.py evaluate
```

### Method 3: Running Individual Components Directly

If you need more control, you can run individual components directly:

1. Train the model using main.py:
```bash
python main.py --dataset_path ../RAVDESS/dataset --epochs 100 --batch_size 32
```

2. Run the web application directly:
```bash
python -m src.webapp
```

## Model Architecture

The enhanced model uses a sophisticated CNN-LSTM architecture:
- Four convolutional layers with increasing filters (64, 128, 256, 512)
- Batch normalization for stable training
- Dropout layers for regularization
- A global pooling branch for overall audio characteristics
- A bidirectional LSTM branch for temporal dependencies
- Feature fusion from both branches before final classification
- Dense layers with proper regularization

## Training Features

- Early stopping to prevent overfitting
- Learning rate scheduling with warm-up and decay
- Model checkpointing to save the best-performing model
- Training resumption capability
- Comprehensive data augmentation for challenging emotions
- Balanced batch sampling

## Web Application

Once the web application is running, open your browser and navigate to:
```
http://localhost:8080
```

The web application provides three main functionalities:
1. **Record Tab**: Record your voice and get real-time emotion prediction
2. **Upload Tab**: Upload audio files for emotion analysis
3. **Live Tab**: Continuous emotion detection from microphone input

## Deployment

The model can be converted to TensorFlow Lite format for deployment on edge devices:
```bash
python main.py --convert_tflite
```

## Model Performance

The enhanced model architecture aims to achieve 99% accuracy on the RAVDESS dataset through:
- Advanced CNN-LSTM architecture with feature fusion
- Comprehensive data augmentation techniques
- Learning rate scheduling and early stopping
- Regularization to prevent overfitting

Detailed results and visualizations are generated during training and can be found in the `logs` and `plots` directories.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 