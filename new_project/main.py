import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import argparse
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.augmentation import AudioAugmenter
from src.training.model_enhanced import EnhancedEmotionModel
from src.evaluation.evaluator import ModelEvaluator
from src.utils.visualization import Visualizer

def main(args):
    """
    Main function to run the emotion recognition training pipeline
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Initialize components
    print("\n=== Initializing Components ===")
    audio_processor = AudioProcessor(
        sample_rate=args.sample_rate,
        duration=args.duration
    )
    
    augmenter = AudioAugmenter(seed=42)
    visualizer = Visualizer(plots_dir=args.plots_dir)
    
    # Prepare data
    print("\n=== Preparing Data ===")
    
    # Augment challenging emotions if enabled
    if args.augment_data:
        print("\nAugmenting challenging emotions...")
        augmenter.augment_dataset(
            args.dataset_path,
            target_emotions=args.target_emotions,
            num_augmentations=args.num_augmentations
        )
    
    # Process dataset and split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = audio_processor.prepare_data(
        args.dataset_path,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42
    )
    
    # Prepare model input
    X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot = \
        audio_processor.prepare_model_input(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save label encoder for inference
    audio_processor.save_label_encoder(
        os.path.join(args.model_dir, 'label_classes.npy')
    )
    
    # Initialize model
    print("\n=== Initializing Enhanced Model ===")
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(audio_processor.label_encoder.classes_)
    
    model = EnhancedEmotionModel(
        input_shape=input_shape,
        num_classes=num_classes,
        model_name=args.model_name
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    print("\n=== Training Enhanced Model ===")
    history = model.train(
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir
    )
    
    # Evaluate model
    print("\n=== Evaluating Model ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model,
        audio_processor.label_encoder,
        output_dir=args.logs_dir
    )
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Generate evaluation metrics
    evaluator.generate_metrics(y_test, y_pred_classes)
    
    # Visualize results
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(
        y_test, 
        y_pred_classes, 
        audio_processor.label_encoder.classes_
    )
    visualizer.plot_classification_report(
        y_test, 
        y_pred_classes, 
        audio_processor.label_encoder.classes_
    )
    
    # Print summary
    visualizer.print_summary()
    
    # Save model
    print("\n=== Saving Model ===")
    h5_path, saved_model_path = model.save_model(
        args.model_dir,
        model_name=args.model_name
    )
    
    # Convert to TFLite if requested
    if args.convert_tflite:
        tflite_path = model.convert_to_tflite(
            args.model_dir,
            model_name=f"{args.model_name}_tflite",
            quantize=True
        )
    
    print(f"\n=== Training Complete ===")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {h5_path}")
    print(f"SavedModel saved to: {saved_model_path}")
    if args.convert_tflite:
        print(f"TFLite model saved to: {tflite_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    
    # Dataset parameters
    parser.add_argument('--dataset_path', type=str, default='RAVDESS/dataset',
                        help='Path to RAVDESS dataset')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, 
                        default=f'emotion_recognition_model_{int(datetime.now().timestamp())}',
                        help='Name of the model')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--plots_dir', type=str, default='plots',
                        help='Directory to save plots')
    
    # Audio parameters
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Audio sample rate')
    parser.add_argument('--duration', type=float, default=3,
                        help='Audio duration in seconds')
    
    # Training parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Validation set size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    
    # Augmentation parameters
    parser.add_argument('--augment_data', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--target_emotions', type=str, nargs='+', 
                        default=['01', '02', '03', '04', '05', '06', '07', '08'],
                        help='Target emotions for augmentation')
    parser.add_argument('--num_augmentations', type=int, default=3,
                        help='Number of augmentations per file')
    
    # Conversion parameters
    parser.add_argument('--convert_tflite', action='store_true', default=True,
                        help='Convert model to TFLite')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
