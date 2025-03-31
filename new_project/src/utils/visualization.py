import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os
import librosa
import librosa.display
import tensorflow as tf

class Visualizer:
    """
    Class for visualizing model performance, training metrics, and audio features.
    Provides comprehensive visualization tools for analysis.
    """
    def __init__(self, plots_dir='plots'):
        """
        Initialize the visualizer
        
        Args:
            plots_dir: Directory to save plots
        """
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.style.use('ggplot')
        
        # Create plots directory if it doesn't exist
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_training_history(self, history):
        """
        Plot training and validation metrics
        
        Args:
            history: Training history from model.fit()
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=16, pad=20)
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Accuracy', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Add horizontal line at 0.99 (99% accuracy)
        ax1.axhline(y=0.99, color='r', linestyle='--', alpha=0.7)
        ax1.text(len(history.history['accuracy'])*0.02, 0.991, '99% Accuracy', 
                fontsize=12, color='r', alpha=0.7)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=16, pad=20)
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Loss', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Add learning rate if available
        if 'lr' in history.history:
            ax3 = ax2.twinx()
            ax3.plot(history.history['lr'], label='Learning Rate', color='g', linestyle=':', linewidth=2)
            ax3.set_ylabel('Learning Rate', fontsize=14)
            ax3.tick_params(axis='y', labelcolor='g')
            ax3.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class names
        """
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        # Create heatmap with improved aesthetics
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
        
        # Add title and labels
        plt.title('Confusion Matrix for Emotion Recognition', fontsize=18, pad=20)
        plt.xlabel('Predicted Emotion', fontsize=16, labelpad=10)
        plt.ylabel('True Emotion', fontsize=16, labelpad=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add text with overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.5, 0.01, f'Overall Accuracy: {accuracy:.4f}', 
                   ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, classes):
        """
        Plot classification report as heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            classes: Class names
        """
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create heatmap with improved aesthetics
        sns.heatmap(report_df.iloc[:-3, :3].astype(float), annot=True, cmap='YlGnBu', 
                   fmt='.3f', annot_kws={"size": 12})
        
        # Add title
        plt.title('Classification Report for Emotion Recognition', fontsize=18, pad=20)
        plt.xlabel('Metrics', fontsize=16, labelpad=10)
        plt.ylabel('Emotions', fontsize=16, labelpad=10)
        
        # Adjust tick labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'classification_report.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distribution(self, features, labels):
        """
        Plot distribution of features by emotion
        
        Args:
            features: Feature matrix
            labels: Emotion labels
        """
        # Reshape features if needed
        if len(features.shape) > 2:
            features = features.reshape(features.shape[0], -1)
        
        # Create figure
        plt.figure(figsize=(16, 6))
        
        # Plot first three features
        for i in range(min(3, features.shape[1])):
            plt.subplot(1, 3, i+1)
            sns.boxplot(x=labels, y=features[:, i])
            plt.title(f'Distribution of Feature {i+1}', fontsize=14)
            plt.xlabel('Emotion', fontsize=12)
            plt.ylabel(f'Feature {i+1} Value', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_mel_spectrogram(self, audio, sr, title='Mel Spectrogram'):
        """
        Plot mel spectrogram of audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
        """
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot spectrogram
        img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.colorbar(img, format='%+2.0f dB')
        plt.title(title, fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Frequency (Hz)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'mel_spectrogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_waveform(self, audio, sr, title='Audio Waveform'):
        """
        Plot waveform of audio
        
        Args:
            audio: Audio signal
            sr: Sample rate
            title: Plot title
        """
        # Create figure
        plt.figure(figsize=(12, 4))
        
        # Plot waveform
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title, fontsize=16)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Amplitude', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'waveform.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_emotion_distribution(self, labels, title='Emotion Distribution'):
        """
        Plot distribution of emotions in dataset
        
        Args:
            labels: Emotion labels
            title: Plot title
        """
        # Count emotions
        emotion_counts = pd.Series(labels).value_counts().sort_index()
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot bar chart
        ax = sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
        
        # Add count labels on top of bars
        for i, count in enumerate(emotion_counts.values):
            ax.text(i, count + 5, str(count), ha='center', fontsize=12)
        
        # Add title and labels
        plt.title(title, fontsize=16)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'emotion_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_architecture(self, model):
        """
        Plot model architecture
        
        Args:
            model: Keras model
        """
        # Create file path
        file_path = os.path.join(self.plots_dir, 'model_architecture.png')
        
        # Plot model
        tf.keras.utils.plot_model(
            model,
            to_file=file_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        
        print(f"Model architecture saved to: {file_path}")
    
    def print_summary(self):
        """Print summary of generated visualizations"""
        print("\nGenerated Visualizations:")
        print("------------------------")
        print(f"1. Training History: {os.path.join(self.plots_dir, 'training_history.png')}")
        print(f"2. Confusion Matrix: {os.path.join(self.plots_dir, 'confusion_matrix.png')}")
        print(f"3. Classification Report: {os.path.join(self.plots_dir, 'classification_report.png')}")
        print(f"4. Feature Distribution: {os.path.join(self.plots_dir, 'feature_distribution.png')}")
        print(f"5. Emotion Distribution: {os.path.join(self.plots_dir, 'emotion_distribution.png')}")
        print(f"6. Model Architecture: {os.path.join(self.plots_dir, 'model_architecture.png')}")
        print("\nAll plots have been saved in the '{self.plots_dir}' directory.")
