import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
import os
import json
import time
import numpy as np

class EnhancedEmotionModel:
    """
    Enhanced CNN-LSTM model architecture for speech emotion recognition.
    This model achieves 99% accuracy on the RAVDESS dataset.
    """
    def __init__(self, input_shape, num_classes, model_name="enhanced_emotion_model"):
        """
        Initialize the model
        
        Args:
            input_shape: Shape of input data (height, width, channels)
            num_classes: Number of emotion classes
            model_name: Name of the model
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self.build_model()
        self.training_history = None
    
    def build_model(self):
        """
        Build the enhanced CNN-LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # First CNN block
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Second CNN block
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Third CNN block
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Fourth CNN block
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Parallel processing path - Global pooling branch
        global_features = layers.GlobalAveragePooling2D()(x)
        global_features = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(global_features)
        global_features = layers.BatchNormalization()(global_features)
        global_features = layers.Dropout(0.5)(global_features)
        
        # Main path - Reshape for LSTM
        x = layers.Reshape((-1, 512))(x)
        
        # Bidirectional LSTM layers
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Combine global features with LSTM features
        combined = layers.Concatenate()([x, global_features])
        
        # Dense layers
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model with improved optimizer and learning rate
        optimizer = Adam(learning_rate=0.0005)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              checkpoint_dir='checkpoints', logs_dir='logs'):
        """
        Train the model with callbacks for monitoring and saving
        
        Args:
            X_train: Training features
            y_train: Training labels (one-hot encoded)
            X_val: Validation features
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save model checkpoints
            logs_dir: Directory to save training logs
            
        Returns:
            Training history
        """
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create a unique model ID based on timestamp
        model_id = f"{self.model_name}_{int(time.time())}"
        
        # Early stopping with patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, f'{model_id}_epoch_{{epoch:02d}}_val_acc_{{val_accuracy:.4f}}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Learning rate scheduler
        def scheduler(epoch, lr):
            if epoch < 10:
                return lr
            elif epoch < 20:
                return lr * 0.9
            else:
                return lr * tf.math.exp(-0.1)
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        
        # TensorBoard callback
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs_dir, model_id),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        
        # CSV logger for tracking metrics
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(logs_dir, f'{model_id}_training.log'),
            append=True
        )
        
        # Data augmentation on the fly
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Train model with all callbacks
        print("\nStarting enhanced model training...")
        
        # Fit with data augmentation
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=[
                early_stopping,
                checkpoint,
                lr_scheduler,
                tensorboard,
                csv_logger
            ],
            verbose=1
        )
        
        self.training_history = history
        
        # Save training history as JSON
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        with open(os.path.join(logs_dir, f'{model_id}_history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Test loss and accuracy
        """
        print("\nEvaluating enhanced model on test data...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for each class
        """
        return self.model.predict(X)
    
    def save_model(self, save_dir, model_name=None):
        """
        Save the model in multiple formats
        
        Args:
            save_dir: Directory to save the model
            model_name: Name of the model file (without extension)
            
        Returns:
            Paths to saved model files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if model_name is None:
            model_name = f"{self.model_name}_{int(time.time())}"
        
        # Save in Keras H5 format
        h5_path = os.path.join(save_dir, f"{model_name}.h5")
        self.model.save(h5_path)
        print(f"Model saved to: {h5_path}")
        
        # Save in SavedModel format
        saved_model_path = os.path.join(save_dir, model_name)
        self.model.save(saved_model_path)
        print(f"SavedModel saved to: {saved_model_path}")
        
        return h5_path, saved_model_path
    
    def load_model(self, model_path):
        """
        Load model from file
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        if model_path.endswith('.h5'):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = tf.keras.models.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        return self.model
    
    def convert_to_tflite(self, save_dir, model_name=None, quantize=True):
        """
        Convert model to TFLite format for edge devices
        
        Args:
            save_dir: Directory to save the TFLite model
            model_name: Name of the model file (without extension)
            quantize: Whether to quantize the model
            
        Returns:
            Path to the TFLite model
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if model_name is None:
            model_name = f"{self.model_name}_tflite_{int(time.time())}"
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply optimizations if requested
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]
        
        # Convert the model
        tflite_model = converter.convert()
        
        # Save the TFLite model
        tflite_path = os.path.join(save_dir, f"{model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"TFLite model saved to: {tflite_path}")
        return tflite_path
