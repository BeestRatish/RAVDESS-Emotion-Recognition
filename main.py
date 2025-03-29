import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import AudioPreprocessor
from model import EmotionRecognitionModel
from visualization import Visualizer
from data_augmentation import AudioAugmentor
import tensorflow as tf

def convert_to_tflite(model, model_save_path):
    """Convert model to TFLite format for edge devices""" 
    # Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the TFLite model
    tflite_model_path = os.path.join(model_save_path, 'emotion_model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {tflite_model_path}")
    
    # Save label encoder classes for inference
    np.save(os.path.join(model_save_path, 'label_classes.npy'), label_encoder.classes_)
    print("Label classes saved for inference")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file"""
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
    if not checkpoints:
        return None
    return os.path.join(checkpoint_dir, max(checkpoints))

def main():
    # Initialize components
    preprocessor = AudioPreprocessor()
    visualizer = Visualizer()
    augmentor = AudioAugmentor()
    
    # Set paths
    dataset_path = "RAVDESS/dataset"
    model_save_path = "models"
    checkpoint_dir = "checkpoints"
    
    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    # Prepare data
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Identify challenging emotions (those with lower performance)
    challenging_emotions = ['06', '08']  # Fearful and Surprised
    
    print("Augmenting challenging emotions...")
    # Augment challenging emotions
    augmented_files = augmentor.augment_dataset(
        dataset_path,
        target_emotions=challenging_emotions,
        num_augmentations=2
    )
    
    # Reload data with augmented samples
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(dataset_path)
    
    # Re-encode labels after augmentation
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Convert labels to one-hot encoding
    y_train_onehot = tf.keras.utils.to_categorical(y_train_encoded, num_classes=len(label_encoder.classes_))
    y_test_onehot = tf.keras.utils.to_categorical(y_test_encoded, num_classes=len(label_encoder.classes_))
    
    # Reshape input for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Print data shapes for verification
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train_onehot.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test_onehot.shape}")
    
    # Initialize model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(label_encoder.classes_)
    model = EmotionRecognitionModel(input_shape, num_classes)
    
    # Check for existing checkpoints
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Found checkpoint: {latest_checkpoint}")
        model.load_checkpoint(latest_checkpoint)
    
    print("Training model...")
    try:
        # Train model
        history = model.train(
            X_train, y_train_onehot,
            X_test, y_test_onehot,
            epochs=50,
            batch_size=32
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        # Save the current model state
        model.model.save(os.path.join(model_save_path, 'interrupted_model.h5'))
        print("Model saved. You can resume training later.")
        return
    
    print("Evaluating model...")
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("Generating visualizations...")
    # Generate visualizations
    visualizer.plot_training_history(history)
    visualizer.plot_confusion_matrix(y_test_encoded, y_pred_classes, label_encoder.classes_)
    visualizer.plot_classification_report(y_test_encoded, y_pred_classes, label_encoder.classes_)
    visualizer.plot_feature_distribution(X_train.reshape(X_train.shape[0], -1), y_train)
    
    # Print visualization summary
    visualizer.print_summary()
    
    # Save model in multiple formats
    print("\nSaving models...")
    # Save Keras model
    model.model.save(os.path.join(model_save_path, 'final_model.h5'))
    print("Keras model saved successfully!")
    
    # Save label encoder classes
    np.save(os.path.join(model_save_path, 'label_classes.npy'), label_encoder.classes_)
    print("Label encoder classes saved successfully!")
    
    # Convert and save TFLite model
    convert_to_tflite(model.model, model_save_path)
    
    print("\nModel deployment files are ready!")

if __name__ == "__main__":
    main() 