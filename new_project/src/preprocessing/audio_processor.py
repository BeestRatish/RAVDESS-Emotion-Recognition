import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class AudioProcessor:
    """
    Class for audio preprocessing and feature extraction from the RAVDESS dataset.
    Handles loading, feature extraction, and data preparation for model training.
    """
    def __init__(self, sample_rate=22050, duration=3, n_mfcc=40, n_mels=40, n_chroma=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_chroma = n_chroma
        self.target_length = 180  # Fixed length for all features
        
        # Emotion mapping
        self.emotions = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, file_path):
        """
        Extract audio features from a file:
        - MFCCs (Mel-frequency cepstral coefficients)
        - Mel spectrogram
        - Chroma features
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Extracted features as a numpy array
        """
        try:
            # Load audio file with fixed duration
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Add padding if audio is too short
            if len(audio) < self.sample_rate * self.duration:
                padding = self.sample_rate * self.duration - len(audio)
                audio = np.pad(audio, (0, padding), 'constant')
            
            # Extract features with fixed sizes
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=self.n_chroma)
            
            # Ensure all features have the same time dimension
            mfcc = librosa.util.fix_length(mfcc, size=self.target_length)
            mel_spec = librosa.util.fix_length(mel_spec, size=self.target_length)
            chroma = librosa.util.fix_length(chroma, size=self.target_length)
            
            # Combine features
            features = np.concatenate([mfcc, mel_spec, chroma], axis=0)
            
            # Add additional features for improved accuracy
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr = librosa.util.fix_length(zcr, size=self.target_length)
            
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_centroid = librosa.util.fix_length(spectral_centroid, size=self.target_length)
            
            # Root mean square energy
            rms = librosa.feature.rms(y=audio)
            rms = librosa.util.fix_length(rms, size=self.target_length)
            
            # Combine all features
            features = np.concatenate([features, zcr, spectral_centroid, rms], axis=0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_dataset(self, dataset_path):
        """
        Process the entire dataset and extract features and labels
        
        Args:
            dataset_path: Path to the RAVDESS dataset
            
        Returns:
            features: Extracted features as numpy array
            labels: Corresponding emotion labels
        """
        features = []
        labels = []
        file_paths = []
        
        print(f"Processing dataset from {dataset_path}...")
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract emotion from filename
                    try:
                        emotion_code = file.split('-')[2]
                        emotion = self.emotions[emotion_code]
                        
                        # Extract features
                        feature = self.extract_features(file_path)
                        
                        if feature is not None:
                            features.append(feature)
                            labels.append(emotion)
                            file_paths.append(file_path)
                    except (IndexError, KeyError) as e:
                        print(f"Error extracting emotion from {file}: {str(e)}")
        
        print(f"Processed {len(features)} audio files successfully.")
        return np.array(features), np.array(labels), file_paths
    
    def prepare_data(self, dataset_path, test_size=0.2, val_size=0.1, random_state=42):
        """
        Prepare train, validation, and test data
        
        Args:
            dataset_path: Path to the RAVDESS dataset
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_val, X_test: Features for training, validation, and testing
            y_train, y_val, y_test: Labels for training, validation, and testing
        """
        # Process dataset
        features, labels, _ = self.process_dataset(dataset_path)
        
        # Fit label encoder
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # First split: training + validation and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=random_state, stratify=encoded_labels
        )
        
        # Second split: training and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state, stratify=y_train_val
        )
        
        # Print data distribution
        print("\nData distribution:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Print class distribution
        train_class_dist = np.bincount(y_train)
        val_class_dist = np.bincount(y_val)
        test_class_dist = np.bincount(y_test)
        
        print("\nClass distribution (training):")
        for i, count in enumerate(train_class_dist):
            print(f"  {self.label_encoder.inverse_transform([i])[0]}: {count}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_model_input(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Prepare input for the model:
        - Reshape features for CNN input
        - Convert labels to one-hot encoding
        
        Args:
            X_train, X_val, X_test: Features for training, validation, and testing
            y_train, y_val, y_test: Labels for training, validation, and testing
            
        Returns:
            Processed data ready for model input
        """
        # Reshape input for CNN (samples, height, width, channels)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        
        # Convert labels to one-hot encoding
        num_classes = len(self.label_encoder.classes_)
        y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)
        y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
        
        print(f"\nInput shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train_onehot.shape}")
        print(f"y_val: {y_val_onehot.shape}")
        print(f"y_test: {y_test_onehot.shape}")
        
        return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot
    
    def save_label_encoder(self, save_path):
        """Save the label encoder classes for inference"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.label_encoder.classes_)
        print(f"Label encoder classes saved to: {save_path}")
    
    def load_label_encoder(self, load_path):
        """Load the label encoder classes for inference"""
        classes = np.load(load_path, allow_pickle=True)
        self.label_encoder.classes_ = classes
        print(f"Label encoder classes loaded from: {load_path}")
        return self.label_encoder
