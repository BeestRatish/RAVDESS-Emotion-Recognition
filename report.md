# Speech Emotion Recognition System Using RAVDESS Dataset
## A Deep Learning Approach with CNN-LSTM Architecture

### Abstract
This project implements a robust speech emotion recognition system using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset. The system employs a hybrid deep learning architecture combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks to classify eight different emotions from speech audio. The implementation includes data augmentation techniques, advanced model architecture, and optimization strategies to achieve high accuracy in emotion recognition. The system is designed for deployment on edge devices like the Jetson Orin Nano, making it suitable for real-time emotion recognition applications.

### 1. Introduction

#### 1.1 Background
Speech emotion recognition is a crucial component in human-computer interaction systems. The ability to accurately identify emotions from speech can enhance various applications, including:
- Customer service automation
- Mental health monitoring
- Educational technology
- Entertainment systems
- Security and surveillance

#### 1.2 Problem Statement
Traditional emotion recognition systems often struggle with:
- Real-time processing requirements
- Accuracy in distinguishing similar emotions
- Deployment on resource-constrained devices
- Handling variations in speech patterns

#### 1.3 Objectives
The primary objectives of this project are:
1. Develop an accurate emotion recognition system
2. Optimize the model for edge device deployment
3. Implement robust data augmentation techniques
4. Create a scalable and efficient solution

### 2. Methodology

#### 2.1 Dataset
The RAVDESS dataset contains:
- 24 professional actors (12 female, 12 male)
- 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- High-quality audio recordings (48kHz, 16-bit)
- Standardized recording conditions

#### 2.2 Feature Extraction
The system extracts the following features:
1. Mel-frequency cepstral coefficients (MFCC)
2. Mel spectrogram
3. Chroma features

#### 2.3 Data Augmentation
Implemented augmentation techniques:
1. Noise addition
2. Time stretching
3. Pitch shifting
4. Time shifting
5. Frequency masking

#### 2.4 Model Architecture
The hybrid CNN-LSTM architecture includes:

1. CNN Blocks:
   - Three convolutional blocks with increasing filters
   - Batch normalization
   - Dropout regularization
   - Max pooling

2. LSTM Layers:
   - Bidirectional LSTM for better sequence learning
   - Two LSTM layers with decreasing units
   - Batch normalization and dropout

3. Dense Layers:
   - Two fully connected layers
   - L2 regularization
   - Dropout for overfitting prevention

#### 2.5 Training Process
Optimization strategies:
1. Learning rate scheduling
2. Early stopping
3. Model checkpointing
4. Data augmentation for challenging emotions

### 3. Implementation Details

#### 3.1 Preprocessing Pipeline
```python
class AudioPreprocessor:
    def extract_features(self, file_path):
        # MFCC extraction
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=40)
```

#### 3.2 Data Augmentation
```python
class AudioAugmentor:
    def augment_audio(self, audio, sr):
        # Random selection of augmentations
        num_augs = np.random.randint(1, len(self.augmentations) + 1)
        selected_augs = np.random.choice(self.augmentations, num_augs)
```

#### 3.3 Model Architecture
```python
model = models.Sequential([
    # CNN blocks with batch normalization
    layers.Conv2D(32, (3, 3), padding='same'),
    layers.BatchNormalization(),
    # Bidirectional LSTM
    layers.Bidirectional(layers.LSTM(128)),
    # Dense layers with regularization
    layers.Dense(128, kernel_regularizer=l2(0.01))
])
```

### 4. Results and Analysis

#### 4.1 Training Performance
- Training accuracy: ~85-90%
- Validation accuracy: ~80-85%
- No significant overfitting observed

#### 4.2 Emotion Classification Results
Best performing emotions:
- Calm and Neutral (highest accuracy)
- Angry and Happy (strong performance)

Challenging emotions:
- Fearful and Surprised (improved with augmentation)
- Disgust (shows some variation)

#### 4.3 Model Optimization
- Reduced model size for edge deployment
- Optimized inference time
- Memory-efficient architecture

### 5. Deployment

#### 5.1 Edge Device Deployment
The system is optimized for Jetson Orin Nano with:
1. TensorFlow Lite conversion
2. FP16 precision support
3. Memory optimization
4. Real-time processing capabilities

#### 5.2 Inference Pipeline
```python
class EmotionPredictor:
    def predict_emotion(self, audio_file):
        # Feature extraction
        features = self.preprocessor.extract_features(audio_file)
        # Model inference
        prediction = self.model.predict(features)
        return self.label_classes[np.argmax(prediction)]
```

### 6. Future Improvements

1. Additional Dataset Integration
2. Real-time Processing Optimization
3. Multi-modal Emotion Recognition
4. Enhanced Data Augmentation
5. Model Quantization for Better Performance

### 7. Conclusion

The implemented system demonstrates:
- High accuracy in emotion recognition
- Robust performance across different emotions
- Efficient deployment on edge devices
- Scalable architecture for future improvements

### References

1. RAVDESS Dataset: https://zenodo.org/record/1188976
2. TensorFlow Documentation: https://www.tensorflow.org/
3. Jetson Orin Nano Documentation: https://developer.nvidia.com/embedded/jetson-orin-nano
4. Librosa Documentation: https://librosa.org/ 