import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.audio_processor import AudioProcessor
from preprocessing.augmentation import AudioAugmenter
from training.model import EmotionRecognitionModel
from evaluation.evaluator import ModelEvaluator
from utils.visualization import Visualizer

class EmotionModelTrainer:
    """
    Class for training the emotion recognition model.
    Handles the entire training pipeline from data preparation to model evaluation.
    """
    def __init__(self, config):
        """
        Initialize the trainer
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['model_dir'], exist_ok=True)
        os.makedirs(config['logs_dir'], exist_ok=True)
        
        # Initialize components
        self.audio_processor = AudioProcessor(
            sample_rate=config['sample_rate'],
            duration=config['duration']
        )
        
        self.augmenter = AudioAugmenter(seed=config['seed'])
        self.visualizer = Visualizer(plots_dir=config['plots_dir'])
        
        # Set random seeds for reproducibility
        np.random.seed(config['seed'])
        tf.random.set_seed(config['seed'])
        
        # Initialize model and evaluator later after data preparation
        self.model = None
        self.evaluator = None
    
    def prepare_data(self):
        """
        Prepare data for training
        
        Returns:
            Processed data ready for model input
        """
        print("\n=== Preparing Data ===")
        
        # Check if augmentation is enabled
        if self.config['augment_data']:
            print("\nAugmenting challenging emotions...")
            self.augmenter.augment_dataset(
                self.config['dataset_path'],
                target_emotions=self.config['target_emotions'],
                num_augmentations=self.config['num_augmentations']
            )
        
        # Process dataset and split into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test = self.audio_processor.prepare_data(
            self.config['dataset_path'],
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['seed']
        )
        
        # Prepare model input
        X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot = \
            self.audio_processor.prepare_model_input(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Save label encoder for inference
        self.audio_processor.save_label_encoder(
            os.path.join(self.config['model_dir'], 'label_classes.npy')
        )
        
        return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot, y_test
    
    def initialize_model(self, input_shape):
        """
        Initialize the model
        
        Args:
            input_shape: Shape of input data
        """
        print("\n=== Initializing Model ===")
        
        # Get number of classes
        num_classes = len(self.audio_processor.label_encoder.classes_)
        
        # Initialize model
        self.model = EmotionRecognitionModel(
            input_shape=input_shape,
            num_classes=num_classes,
            model_name=self.config['model_name']
        )
        
        # Print model summary
        self.model.summary()
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels (one-hot encoded)
            
        Returns:
            Training history
        """
        print("\n=== Training Model ===")
        
        # Train model
        history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            checkpoint_dir=self.config['checkpoint_dir'],
            logs_dir=self.config['logs_dir']
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test, y_test_raw):
        """
        Evaluate the model
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            y_test_raw: Test labels (not one-hot encoded)
        """
        print("\n=== Evaluating Model ===")
        
        # Evaluate model
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(
            self.model,
            self.audio_processor.label_encoder
        )
        
        # Generate predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate evaluation metrics and visualizations
        self.evaluator.generate_metrics(y_test_raw, y_pred_classes)
        
        # Visualize results
        self.visualizer.plot_training_history(self.model.training_history)
        self.visualizer.plot_confusion_matrix(
            y_test_raw, 
            y_pred_classes, 
            self.audio_processor.label_encoder.classes_
        )
        self.visualizer.plot_classification_report(
            y_test_raw, 
            y_pred_classes, 
            self.audio_processor.label_encoder.classes_
        )
        
        # Print summary
        self.visualizer.print_summary()
    
    def save_model(self):
        """Save the model in multiple formats"""
        print("\n=== Saving Model ===")
        
        # Save Keras model
        h5_path, saved_model_path = self.model.save_model(
            self.config['model_dir'],
            model_name=self.config['model_name']
        )
        
        # Convert to TFLite
        if self.config['convert_tflite']:
            tflite_path = self.model.convert_to_tflite(
                self.config['model_dir'],
                model_name=f"{self.config['model_name']}_tflite",
                quantize=True
            )
    
    def run(self):
        """Run the entire training pipeline"""
        start_time = datetime.now()
        print(f"\nStarting training pipeline at {start_time}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = self.prepare_data()
        
        # Initialize model
        self.initialize_model(input_shape=(X_train.shape[1], X_train.shape[2], 1))
        
        # Train model
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        self.evaluate_model(X_test, y_test, y_test_raw)
        
        # Save model
        self.save_model()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        print(f"\nTraining pipeline completed in {training_time}")
        print(f"Model and artifacts saved to: {self.config['model_dir']}")


def get_default_config():
    """Get default configuration"""
    return {
        'dataset_path': 'data/RAVDESS',
        'model_name': f'emotion_recognition_model_{int(datetime.now().timestamp())}',
        'model_dir': 'models',
        'checkpoint_dir': 'checkpoints',
        'logs_dir': 'logs',
        'plots_dir': 'plots',
        'sample_rate': 22050,
        'duration': 3,
        'test_size': 0.2,
        'val_size': 0.1,
        'epochs': 50,
        'batch_size': 32,
        'seed': 42,
        'augment_data': True,
        'target_emotions': ['06', '08'],  # Fearful and Surprised
        'num_augmentations': 2,
        'convert_tflite': True
    }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    
    parser.add_argument('--dataset_path', type=str, help='Path to RAVDESS dataset')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--model_dir', type=str, help='Directory to save models')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory to save checkpoints')
    parser.add_argument('--logs_dir', type=str, help='Directory to save logs')
    parser.add_argument('--plots_dir', type=str, help='Directory to save plots')
    parser.add_argument('--sample_rate', type=int, help='Audio sample rate')
    parser.add_argument('--duration', type=float, help='Audio duration in seconds')
    parser.add_argument('--test_size', type=float, help='Test set size')
    parser.add_argument('--val_size', type=float, help='Validation set size')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--augment_data', action='store_true', help='Enable data augmentation')
    parser.add_argument('--no_augment_data', action='store_false', dest='augment_data', help='Disable data augmentation')
    parser.add_argument('--target_emotions', type=str, nargs='+', help='Target emotions for augmentation')
    parser.add_argument('--num_augmentations', type=int, help='Number of augmentations per file')
    parser.add_argument('--convert_tflite', action='store_true', help='Convert model to TFLite')
    parser.add_argument('--no_convert_tflite', action='store_false', dest='convert_tflite', help='Do not convert model to TFLite')
    
    args = parser.parse_args()
    
    # Get default config
    config = get_default_config()
    
    # Update config with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None:
            config[arg] = getattr(args, arg)
    
    return config


if __name__ == '__main__':
    # Parse arguments
    config = parse_arguments()
    
    # Create trainer
    trainer = EmotionModelTrainer(config)
    
    # Run training pipeline
    trainer.run()
