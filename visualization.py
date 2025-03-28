import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import os

class Visualizer:
    def __init__(self):
        # Set seaborn style
        sns.set_style("whitegrid")
        plt.style.use('default')
        
        # Create plots directory if it doesn't exist
        self.plots_dir = "plots"
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy Over Time', fontsize=12, pad=15)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss Over Time', fontsize=12, pad=15)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=classes, yticklabels=classes)
        
        plt.title('Confusion Matrix for Emotion Recognition', fontsize=14, pad=20)
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_classification_report(self, y_true, y_pred, classes):
        """Plot classification report as heatmap"""
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(report_df.iloc[:-3, :].astype(float), annot=True, cmap='Blues', fmt='.3f')
        
        plt.title('Classification Report for Emotion Recognition', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'classification_report.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distribution(self, features, labels):
        """Plot distribution of features"""
        plt.figure(figsize=(15, 5))
        
        # Plot first three features
        for i in range(min(3, features.shape[1])):
            plt.subplot(1, 3, i+1)
            sns.boxplot(x=labels, y=features[:, i])
            plt.title(f'Distribution of Feature {i+1}', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'feature_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print summary of generated visualizations"""
        print("\nGenerated Visualizations:")
        print("------------------------")
        print(f"1. Training History: {os.path.join(self.plots_dir, 'training_history.png')}")
        print(f"2. Confusion Matrix: {os.path.join(self.plots_dir, 'confusion_matrix.png')}")
        print(f"3. Classification Report: {os.path.join(self.plots_dir, 'classification_report.png')}")
        print(f"4. Feature Distribution: {os.path.join(self.plots_dir, 'feature_distribution.png')}")
        print("\nAll plots have been saved in the 'plots' directory.") 