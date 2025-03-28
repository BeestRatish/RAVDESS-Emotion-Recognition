import librosa
import numpy as np
import soundfile as sf
import os

class AudioAugmentor:
    def __init__(self):
        self.augmentations = [
            self.add_noise,
            self.time_stretch,
            self.pitch_shift,
            self.time_shift,
            self.frequency_mask
        ]
    
    def add_noise(self, audio, sr):
        """Add Gaussian noise to the audio"""
        noise = np.random.normal(0, 0.005, len(audio))
        return audio + noise
    
    def time_stretch(self, audio, sr):
        """Time stretching"""
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, sr):
        """Pitch shifting"""
        n_steps = np.random.randint(-4, 5)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def time_shift(self, audio, sr):
        """Time shifting"""
        shift = np.random.randint(-int(sr * 0.1), int(sr * 0.1))
        return np.roll(audio, shift)
    
    def frequency_mask(self, audio, sr):
        """Frequency masking"""
        # Convert to spectrogram
        spec = librosa.stft(audio)
        # Apply frequency mask
        mask = np.random.rand(*spec.shape) > 0.9
        spec[mask] = 0
        # Convert back to audio
        return librosa.istft(spec)
    
    def augment_audio(self, audio, sr, num_augmentations=1):
        """Apply random augmentations to the audio"""
        augmented_audios = []
        
        for _ in range(num_augmentations):
            aug_audio = audio.copy()
            # Randomly select and apply augmentations
            num_augs = np.random.randint(1, len(self.augmentations) + 1)
            selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
            
            for aug in selected_augs:
                aug_audio = aug(aug_audio, sr)
            
            augmented_audios.append(aug_audio)
        
        return augmented_audios
    
    def augment_dataset(self, dataset_path, target_emotions, num_augmentations=2):
        """Augment specific emotions in the dataset"""
        augmented_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    # Check if this emotion needs augmentation
                    emotion_code = file.split('-')[2]
                    if emotion_code in target_emotions:
                        file_path = os.path.join(root, file)
                        audio, sr = librosa.load(file_path)
                        
                        # Generate augmented versions
                        augmented_audios = self.augment_audio(audio, sr, num_augmentations)
                        
                        # Save augmented files
                        for i, aug_audio in enumerate(augmented_audios):
                            new_filename = f"{os.path.splitext(file)[0]}_aug{i+1}.wav"
                            new_filepath = os.path.join(root, new_filename)
                            sf.write(new_filepath, aug_audio, sr)
                            augmented_files.append(new_filepath)
        
        return augmented_files 