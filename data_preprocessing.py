import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=3):
        self.sample_rate = sample_rate
        self.duration = duration
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
        
    def extract_features(self, file_path):
        """Extract features from audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Extract features with fixed sizes
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=40)
            
            # Ensure all features have the same time dimension
            target_length = 180  # Fixed length for all features
            mfcc = librosa.util.fix_length(mfcc, size=target_length)
            mel_spec = librosa.util.fix_length(mel_spec, size=target_length)
            chroma = librosa.util.fix_length(chroma, size=target_length)
            
            # Combine features
            features = np.concatenate([mfcc, mel_spec, chroma], axis=0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
    
    def process_dataset(self, dataset_path):
        """Process entire dataset and return features and labels"""
        features = []
        labels = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    # Extract emotion from filename
                    emotion_code = file.split('-')[2]
                    emotion = self.emotions[emotion_code]
                    
                    # Extract features
                    feature = self.extract_features(file_path)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion)
        
        return np.array(features), np.array(labels)
    
    def prepare_data(self, dataset_path, test_size=0.2):
        """Prepare train and test data"""
        features, labels = self.process_dataset(dataset_path)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test 