# Speech Emotion Recognition System using RAVDESS Dataset

This project implements a deep learning-based speech emotion recognition system using the RAVDESS dataset. The system uses a CNN-LSTM architecture to classify different emotions from speech audio.

## Features

- Audio preprocessing and feature extraction
- Data augmentation for underrepresented emotions
- CNN-LSTM model architecture with batch normalization and dropout
- Model checkpointing and training resumption
- Visualization of training metrics and model performance
- TensorFlow Lite conversion for edge device deployment
- Comprehensive evaluation metrics

## Project Structure

```
.
├── data_preprocessing.py    # Audio preprocessing and feature extraction
├── data_augmentation.py     # Audio augmentation techniques
├── model.py                # CNN-LSTM model architecture
├── visualization.py        # Visualization utilities
├── main.py                # Main training and evaluation script
├── convert_to_pdf.py      # Report generation script
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
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
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
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

The project uses the RAVDESS dataset. You need to download and place it in the `RAVDESS/dataset` directory.

## Usage

1. Prepare the dataset:
```bash
python data_preprocessing.py
```

2. Train the model:
```bash
python main.py
```

3. Generate visualizations and reports:
```bash
python convert_to_pdf.py
```

## Model Architecture

The model uses a hybrid CNN-LSTM architecture:
- CNN layers for feature extraction
- Batch normalization for stable training
- Dropout layers for regularization
- Bidirectional LSTM layers for temporal dependencies
- Dense layers for final classification

## Training Features

- Early stopping to prevent overfitting
- Learning rate scheduling
- Model checkpointing
- Training resumption capability
- Data augmentation for challenging emotions

## Deployment

The model can be converted to TensorFlow Lite format for deployment on edge devices:
```python
python main.py --convert_tflite
```

## Results

The model achieves competitive performance on the RAVDESS dataset. Detailed results and visualizations are generated during training.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 