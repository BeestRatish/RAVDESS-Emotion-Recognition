import librosa
import numpy as np
import soundfile as sf
import os
import random
from tqdm import tqdm

class AudioAugmenter:
    """
    Class for audio data augmentation to improve model performance.
    Provides various augmentation techniques for audio data.
    """
    def __init__(self, seed=42):
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # List of augmentation techniques
        self.augmentations = [
            self.add_noise,
            self.time_stretch,
            self.pitch_shift,
            self.time_shift,
            self.frequency_mask,
            self.add_reverb,
            self.add_background_noise
        ]
        
        # Create background noise samples
        self._create_background_noise()
    
    def _create_background_noise(self):
        """Create background noise samples for augmentation"""
        # White noise
        self.white_noise = np.random.normal(0, 0.005, int(3 * 22050))
        
        # Pink noise (1/f noise)
        f = np.fft.rfftfreq(len(self.white_noise))
        f[0] = 1  # Avoid division by zero
        pink_spectrum = 1 / f**0.5
        pink_spectrum[0] = 0  # Remove DC component
        phase = np.random.uniform(0, 2*np.pi, len(pink_spectrum))
        pink_spectrum = pink_spectrum * np.exp(1j * phase)
        self.pink_noise = np.fft.irfft(pink_spectrum)
        self.pink_noise = self.pink_noise / np.max(np.abs(self.pink_noise)) * 0.005
        
        # Brown noise (1/f^2 noise)
        brown_spectrum = 1 / f
        brown_spectrum[0] = 0  # Remove DC component
        phase = np.random.uniform(0, 2*np.pi, len(brown_spectrum))
        brown_spectrum = brown_spectrum * np.exp(1j * phase)
        self.brown_noise = np.fft.irfft(brown_spectrum)
        self.brown_noise = self.brown_noise / np.max(np.abs(self.brown_noise)) * 0.005
    
    def add_noise(self, audio, sr):
        """
        Add Gaussian noise to the audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        noise_level = np.random.uniform(0.001, 0.01)
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def time_stretch(self, audio, sr):
        """
        Time stretching (speed up or slow down)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        rate = np.random.uniform(0.8, 1.2)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, sr):
        """
        Pitch shifting (higher or lower pitch)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        n_steps = np.random.randint(-4, 5)
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def time_shift(self, audio, sr):
        """
        Time shifting (shift audio left or right)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        shift = np.random.randint(-int(sr * 0.1), int(sr * 0.1))
        return np.roll(audio, shift)
    
    def frequency_mask(self, audio, sr):
        """
        Frequency masking (mask certain frequency bands)
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        # Convert to spectrogram
        spec = librosa.stft(audio)
        
        # Apply frequency mask
        mask_percentage = np.random.uniform(0.05, 0.15)
        mask = np.random.rand(*spec.shape) > mask_percentage
        spec = spec * mask
        
        # Convert back to audio
        return librosa.istft(spec)
    
    def add_reverb(self, audio, sr):
        """
        Add reverb effect to audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        # Simple reverb implementation using convolution with decaying impulse response
        reverb_length = int(sr * np.random.uniform(0.1, 0.3))
        decay = np.exp(-np.linspace(0, 5, reverb_length))
        impulse_response = np.random.randn(reverb_length) * decay
        
        # Normalize impulse response
        impulse_response = impulse_response / np.sum(np.abs(impulse_response))
        
        # Apply reverb through convolution
        reverb_audio = np.convolve(audio, impulse_response, mode='full')[:len(audio)]
        
        # Mix with original (wet/dry mix)
        mix_ratio = np.random.uniform(0.1, 0.3)
        return (1 - mix_ratio) * audio + mix_ratio * reverb_audio
    
    def add_background_noise(self, audio, sr):
        """
        Add background noise to audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Augmented audio signal
        """
        # Select noise type
        noise_type = np.random.choice(['white', 'pink', 'brown'])
        
        if noise_type == 'white':
            noise = self.white_noise
        elif noise_type == 'pink':
            noise = self.pink_noise
        else:
            noise = self.brown_noise
            
        # Ensure noise is the same length as audio
        if len(noise) > len(audio):
            noise = noise[:len(audio)]
        else:
            # Pad noise if it's shorter
            noise = np.pad(noise, (0, max(0, len(audio) - len(noise))), mode='wrap')
        
        # Mix noise with audio
        noise_level = np.random.uniform(0.01, 0.05)
        return audio + noise_level * noise
    
    def augment_audio(self, audio, sr, num_augmentations=1):
        """
        Apply random augmentations to the audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            num_augmentations: Number of augmented versions to create
            
        Returns:
            List of augmented audio signals
        """
        augmented_audios = []
        
        for _ in range(num_augmentations):
            aug_audio = audio.copy()
            
            # Randomly select and apply augmentations
            num_augs = np.random.randint(1, 4)  # Apply 1-3 augmentations
            selected_augs = np.random.choice(self.augmentations, num_augs, replace=False)
            
            for aug in selected_augs:
                aug_audio = aug(aug_audio, sr)
            
            # Normalize audio after augmentation
            if np.max(np.abs(aug_audio)) > 0:
                aug_audio = aug_audio / np.max(np.abs(aug_audio))
                
            augmented_audios.append(aug_audio)
        
        return augmented_audios
    
    def augment_dataset(self, dataset_path, target_emotions=None, num_augmentations=2, output_dir=None):
        """
        Augment specific emotions in the dataset
        
        Args:
            dataset_path: Path to the dataset
            target_emotions: List of emotion codes to augment (e.g., ['06', '08'])
            num_augmentations: Number of augmented versions to create per file
            output_dir: Directory to save augmented files (if None, save in original location)
            
        Returns:
            List of paths to augmented files
        """
        augmented_files = []
        
        # If no target emotions specified, augment all
        if target_emotions is None:
            target_emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
            
        print(f"Augmenting emotions: {', '.join(target_emotions)}")
        print(f"Creating {num_augmentations} augmented versions per file")
        
        # Find all .wav files
        all_files = []
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    all_files.append((root, file))
        
        # Process files with progress bar
        for root, file in tqdm(all_files, desc="Augmenting audio files"):
            # Check if this emotion needs augmentation
            try:
                emotion_code = file.split('-')[2]
                if emotion_code in target_emotions:
                    file_path = os.path.join(root, file)
                    
                    # Load audio
                    audio, sr = librosa.load(file_path, sr=None)
                    
                    # Generate augmented versions
                    augmented_audios = self.augment_audio(audio, sr, num_augmentations)
                    
                    # Determine output directory
                    if output_dir is None:
                        save_dir = root
                    else:
                        # Recreate directory structure in output_dir
                        rel_path = os.path.relpath(root, dataset_path)
                        save_dir = os.path.join(output_dir, rel_path)
                        os.makedirs(save_dir, exist_ok=True)
                    
                    # Save augmented files
                    for i, aug_audio in enumerate(augmented_audios):
                        new_filename = f"{os.path.splitext(file)[0]}_aug{i+1}.wav"
                        new_filepath = os.path.join(save_dir, new_filename)
                        sf.write(new_filepath, aug_audio, sr)
                        augmented_files.append(new_filepath)
            except (IndexError, KeyError) as e:
                print(f"Error processing {file}: {str(e)}")
        
        print(f"Created {len(augmented_files)} augmented audio files")
        return augmented_files
