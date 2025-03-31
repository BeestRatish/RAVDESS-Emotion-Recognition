import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tqdm import tqdm
import warnings
import random
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """
    Class for audio preprocessing and feature extraction from the RAVDESS dataset.
    Handles loading, feature extraction, and data preparation for model training.
    """
    def __init__(self, sample_rate=22050, duration=3, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512, n_chroma=40):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma
        self.target_length = 216  # Increased fixed length for all features
        
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
        
        # Initialize label encoder and scaler
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Store feature statistics for normalization
        self.feature_mean = None
        self.feature_std = None
        
    def extract_features(self, file_path, augment=False):
        """
        Extract enhanced audio features from a file:
        - MFCCs (Mel-frequency cepstral coefficients) with delta and delta-delta
        - Mel spectrogram
        - Spectral contrast
        - Tonnetz features
        - Zero crossing rate
        - Spectral centroid
        - Root mean square energy
        
        Args:
            file_path: Path to the audio file
            augment: Whether to apply random augmentation
            
        Returns:
            Extracted features as a numpy array
        """
        try:
            # Load audio file with fixed duration
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Apply random augmentation if requested
            if augment:
                audio = self._apply_random_augmentation(audio)
            
            # Ensure audio is of fixed length
            if len(audio) < self.sample_rate * self.duration:
                audio = np.pad(audio, (0, int(self.sample_rate * self.duration) - len(audio)), 'constant')
            else:
                audio = audio[:int(self.sample_rate * self.duration)]
            
            # Extract MFCCs with delta and delta-delta
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Extract Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Extract spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Extract tonnetz features
            harmonic = librosa.effects.harmonic(audio)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            
            # Extract zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)
            
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Extract root mean square energy
            rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
            
            # Ensure all features have the same time dimension
            mfcc = librosa.util.fix_length(mfcc, size=self.target_length)
            delta_mfcc = librosa.util.fix_length(delta_mfcc, size=self.target_length)
            delta2_mfcc = librosa.util.fix_length(delta2_mfcc, size=self.target_length)
            mel_spec_db = librosa.util.fix_length(mel_spec_db, size=self.target_length)
            contrast = librosa.util.fix_length(contrast, size=self.target_length)
            tonnetz = librosa.util.fix_length(tonnetz, size=self.target_length)
            zcr = librosa.util.fix_length(zcr, size=self.target_length)
            spectral_centroid = librosa.util.fix_length(spectral_centroid, size=self.target_length)
            rms = librosa.util.fix_length(rms, size=self.target_length)
            
            # Combine all features
            features = np.concatenate([
                mfcc, delta_mfcc, delta2_mfcc, mel_spec_db, contrast,
                tonnetz, zcr, spectral_centroid, rms
            ], axis=0)
            
            return features
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None
            
    def _apply_random_augmentation(self, audio):
        """
        Apply random augmentation to audio
        
        Args:
            audio: Audio signal
            
        Returns:
            Augmented audio signal
        """
        # Choose random augmentation
        aug_type = random.choice(['pitch', 'speed', 'noise', 'shift', 'stretch', 'none'])
        
        if aug_type == 'pitch':
            # Random pitch shift between -3 and 3 semitones
            n_steps = random.uniform(-3, 3)
            return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        elif aug_type == 'speed':
            # Random speed change between 0.85 and 1.15
            rate = random.uniform(0.85, 1.15)
            return librosa.effects.time_stretch(audio, rate=rate)
        
        elif aug_type == 'noise':
            # Add random noise
            noise_level = random.uniform(0.001, 0.01)
            noise = np.random.randn(len(audio))
            return audio + noise_level * noise
        
        elif aug_type == 'shift':
            # Random time shift
            shift_amount = random.randint(-self.sample_rate // 8, self.sample_rate // 8)
            return np.roll(audio, shift_amount)
            
        elif aug_type == 'stretch':
            # Time stretching without changing pitch
            stretch_factor = random.uniform(0.8, 1.2)
            stretched = librosa.effects.time_stretch(audio, rate=stretch_factor)
            # Ensure length is preserved
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            else:
                stretched = np.pad(stretched, (0, max(0, len(audio) - len(stretched))), 'constant')
            return stretched
        
        else:  # 'none'
            return audio
    
    def process_dataset(self, dataset_path, augment=False, augment_factor=2):
        """
        Process the entire dataset and extract features and labels
        
        Args:
            dataset_path: Path to the RAVDESS dataset
            augment: Whether to apply data augmentation
            augment_factor: How many augmented samples to create per original sample
            
        Returns:
            features: Extracted features as numpy array
            labels: Corresponding emotion labels
            file_paths: Paths to the audio files
        """
        features = []
        labels = []
        file_paths = []
        
        print(f"Processing dataset from {dataset_path}...")
        
        # Get all wav files
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    all_files.append((root, file))
        
        # Process files with progress bar
        for root, file in tqdm(all_files, desc="Extracting features"):
            file_path = os.path.join(root, file)
            
            # Extract emotion from filename
            try:
                emotion_code = file.split('-')[2]
                emotion = self.emotions[emotion_code]
                
                # Extract features for original audio
                feature = self.extract_features(file_path)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)
                    file_paths.append(file_path)
                    
                    # Apply augmentation if requested
                    if augment:
                        for i in range(augment_factor):
                            # Extract features with augmentation
                            aug_feature = self.extract_features(file_path, augment=True)
                            
                            if aug_feature is not None:
                                features.append(aug_feature)
                                labels.append(emotion)
                                file_paths.append(f"{file_path}_aug_{i}")
                
            except (IndexError, KeyError) as e:
                print(f"Error extracting emotion from {file}: {str(e)}")
        
        # Convert to numpy arrays
        features_array = np.array(features)
        labels_array = np.array(labels)
        
        # Normalize features
        if features_array.size > 0:
            # Reshape to 2D for normalization
            orig_shape = features_array.shape
            features_2d = features_array.reshape(orig_shape[0], -1)
            
            # Fit scaler and transform
            self.scaler.fit(features_2d)
            features_2d = self.scaler.transform(features_2d)
            
            # Reshape back to original shape
            features_array = features_2d.reshape(orig_shape)
            
            # Store mean and std for future normalization
            self.feature_mean = self.scaler.mean_
            self.feature_std = self.scaler.scale_
        
        print(f"Processed {len(features)} audio files successfully.")
        print(f"Feature array shape: {features_array.shape}")
        
        # Visualize class distribution
        self._visualize_class_distribution(labels_array)
        
        return features_array, labels_array, file_paths
    
    def _visualize_class_distribution(self, labels):
        """
        Visualize the class distribution
        
        Args:
            labels: Array of emotion labels
        """
        # Count occurrences of each emotion
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique_labels, counts, color='skyblue')
        
        # Add count labels on top of bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    str(count), ha='center', va='bottom')
        
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Emotion')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/class_distribution.png')
        plt.close()
    
    def prepare_data(self, dataset_path, test_size=0.2, val_size=0.1, random_state=42, augment=True):
        """
        Prepare train, validation, and test data
        
        Args:
            dataset_path: Path to the RAVDESS dataset
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            augment: Whether to apply data augmentation
            
        Returns:
            X_train, X_val, X_test: Features for training, validation, and testing
            y_train, y_val, y_test: Labels for training, validation, and testing
        """
        # Process dataset with augmentation
        features, labels, _ = self.process_dataset(dataset_path, augment=augment)
        
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
        print(f"Feature dimensions: {X_train.shape[1:]}")
        
        # Print class distribution
        train_class_dist = np.bincount(y_train)
        val_class_dist = np.bincount(y_val)
        test_class_dist = np.bincount(y_test)
        
        print("\nClass distribution (training):")
        for i, count in enumerate(train_class_dist):
            print(f"  {self.label_encoder.inverse_transform([i])[0]}: {count}")
        
        # Visualize train/val/test split
        self._visualize_data_split(len(X_train), len(X_val), len(X_test))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _visualize_data_split(self, train_size, val_size, test_size):
        """
        Visualize the train/val/test split
        
        Args:
            train_size: Number of training samples
            val_size: Number of validation samples
            test_size: Number of test samples
        """
        # Create pie chart
        plt.figure(figsize=(10, 7))
        sizes = [train_size, val_size, test_size]
        labels = ['Training', 'Validation', 'Testing']
        colors = ['#66b3ff', '#99ff99', '#ffcc99']
        explode = (0.1, 0, 0)  # explode the 1st slice (Training)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                shadow=True, startangle=140, textprops={'fontsize': 14})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title('Dataset Split', fontsize=16)
        
        # Save the plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/dataset_split.png')
        plt.close()
    
    def prepare_model_input(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Prepare input for the model:
        - Reshape features for CNN input
        - Convert labels to one-hot encoding
        - Create efficient TensorFlow datasets
        
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
        
        # Create TensorFlow datasets for efficient loading
        batch_size = 32
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_onehot))
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_onehot))
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_onehot))
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        print(f"\nInput shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train_onehot.shape}")
        print(f"y_val: {y_val_onehot.shape}")
        print(f"y_test: {y_test_onehot.shape}")
        
        # Visualize sample spectrograms
        self._visualize_sample_spectrograms(X_train, y_train, 3)
        
        return train_dataset, val_dataset, test_dataset, X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot
    
    def _visualize_sample_spectrograms(self, X, y, num_samples=3):
        """
        Visualize sample spectrograms for each emotion
        
        Args:
            X: Feature array
            y: Label array
            num_samples: Number of samples to visualize per emotion
        """
        # Create directory for plots
        os.makedirs('plots/spectrograms', exist_ok=True)
        
        # Get unique emotions
        unique_emotions = np.unique(self.label_encoder.inverse_transform(y))
        
        # Visualize samples for each emotion
        for emotion in unique_emotions:
            # Get indices for this emotion
            emotion_idx = np.where(self.label_encoder.inverse_transform(y) == emotion)[0]
            
            # Select random samples
            if len(emotion_idx) > num_samples:
                sample_idx = np.random.choice(emotion_idx, num_samples, replace=False)
            else:
                sample_idx = emotion_idx
            
            # Plot samples
            for i, idx in enumerate(sample_idx):
                plt.figure(figsize=(10, 6))
                
                # Get the mel spectrogram part (assuming it's in the feature stack)
                # The exact slice depends on your feature arrangement
                mel_spec = X[idx, self.n_mfcc*3:self.n_mfcc*3+self.n_mels, :, 0]
                
                # Display spectrogram
                plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Mel Spectrogram - Emotion: {emotion}')
                plt.xlabel('Time')
                plt.ylabel('Mel Frequency')
                plt.tight_layout()
                
                # Save the plot
                plt.savefig(f'plots/spectrograms/{emotion}_{i}.png')
                plt.close()
    
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
