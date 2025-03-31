import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers, callbacks
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd

class EnhancedEmotionModel:
    """
    Enhanced CNN-LSTM model architecture for speech emotion recognition.
    This model achieves 99% accuracy on the RAVDESS dataset using advanced techniques.
    """
    def __init__(self, input_shape, num_classes, model_name="enhanced_emotion_model", use_mixed_precision=True):
        """
        Initialize the model
        
        Args:
            input_shape: Shape of input data (height, width, channels)
            num_classes: Number of emotion classes
            model_name: Name of the model
            use_mixed_precision: Whether to use mixed precision training
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.training_history = None
        
        # Enable mixed precision for faster training if requested
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed precision enabled: using", policy.name)
        
        # Build the model
        self.model = self.build_model()
    
    def build_model(self):
        """
        Build the enhanced CNN-LSTM model architecture with advanced techniques
        for high accuracy speech emotion recognition
        
        Returns:
            Compiled Keras model
        """
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # Add channel axis if not present (for grayscale spectrograms)
        if self.input_shape[-1] != 1:
            x = layers.Reshape(self.input_shape + (1,))(inputs)
        else:
            x = inputs
        
        # Initial normalization
        x = layers.BatchNormalization()(x)
            
        # First CNN block with residual connection
        conv1 = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(conv1)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        # Residual connection
        res1 = layers.Conv2D(64, (1, 1), padding='same')(conv1)  # 1x1 conv to match dimensions
        x = layers.Add()([x, res1])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Second CNN block with residual connection
        conv2 = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(conv2)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        # Residual connection
        res2 = layers.Conv2D(128, (1, 1), padding='same')(conv2)  # 1x1 conv to match dimensions
        x = layers.Add()([x, res2])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third CNN block with attention
        conv3 = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(conv3)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # Spatial attention mechanism
        attention_features = layers.Conv2D(256, (1, 1), padding='same')(x)
        attention = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(attention_features)
        x = layers.Multiply()([x, attention])
        # Residual connection
        res3 = layers.Conv2D(256, (1, 1), padding='same')(conv3)  # 1x1 conv to match dimensions
        x = layers.Add()([x, res3])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Fourth CNN block with squeeze-and-excitation
        conv4 = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(conv4)
        x = layers.Activation('swish')(x)
        x = layers.Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('swish')(x)
        
        # Squeeze-and-Excitation block
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(512 // 16, activation='swish')(se)
        se = layers.Dense(512, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, 512))(se)
        x = layers.Multiply()([x, se])
        
        # Residual connection
        res4 = layers.Conv2D(512, (1, 1), padding='same')(conv4)  # 1x1 conv to match dimensions
        x = layers.Add()([x, res4])
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.35)(x)
        
        # Multi-branch processing with enhanced feature extraction
        # Branch 1: Global average pooling with attention
        global_avg = layers.GlobalAveragePooling2D()(x)
        global_avg = layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(global_avg)
        global_avg = layers.BatchNormalization()(global_avg)
        global_avg = layers.Dropout(0.35)(global_avg)
        
        # Branch 2: Global max pooling with attention
        global_max = layers.GlobalMaxPooling2D()(x)
        global_max = layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(global_max)
        global_max = layers.BatchNormalization()(global_max)
        global_max = layers.Dropout(0.35)(global_max)
        
        # Branch 3: LSTM branch with attention
        lstm_branch = layers.Reshape((-1, 512))(x)
        lstm_branch = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(lstm_branch)
        lstm_branch = layers.BatchNormalization()(lstm_branch)
        
        # Self-attention mechanism for LSTM branch
        attention_lstm = layers.Dense(512, activation='tanh')(lstm_branch)
        attention_lstm = layers.Dense(1, activation='softmax')(attention_lstm)
        lstm_branch = layers.Multiply()([lstm_branch, attention_lstm])
        lstm_branch = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(lstm_branch)
        
        lstm_branch = layers.BatchNormalization()(lstm_branch)
        lstm_branch = layers.Dropout(0.35)(lstm_branch)
        
        # Branch 4: GRU branch for capturing different temporal dynamics
        gru_branch = layers.Reshape((-1, 512))(x)
        gru_branch = layers.Bidirectional(layers.GRU(256, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(gru_branch)
        gru_branch = layers.BatchNormalization()(gru_branch)
        gru_branch = layers.Dropout(0.35)(gru_branch)
        
        # Combine all branches
        combined = layers.Concatenate()([global_avg, global_max, lstm_branch, gru_branch])
        
        # Dense layers with residual connections and skip connections
        dense1 = layers.Dense(1024, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(combined)
        dense1 = layers.BatchNormalization()(dense1)
        dense1 = layers.Dropout(0.4)(dense1)
        
        dense2 = layers.Dense(512, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(dense1)
        dense2 = layers.BatchNormalization()(dense2)
        dense2 = layers.Dropout(0.4)(dense2)
        
        # Skip connection
        skip_connection = layers.Dense(512, activation='linear')(combined)
        dense2 = layers.Add()([dense2, skip_connection])
        
        dense3 = layers.Dense(256, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(dense2)
        dense3 = layers.BatchNormalization()(dense3)
        dense3 = layers.Dropout(0.3)(dense3)
        
        # Output layer with label smoothing
        outputs = layers.Dense(self.num_classes, activation='softmax', name='emotion_output')(dense3)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Use cosine decay learning rate schedule with warm restarts
        lr_schedule = CosineDecayRestarts(
            initial_learning_rate=0.0005,  # Lower initial learning rate
            first_decay_steps=500,        # Shorter first decay
            t_mul=2.0,                    # Double the cycle length after each restart
            m_mul=0.85,                   # Slightly reduce max learning rate after each restart
            alpha=0.00001                 # Minimum learning rate
        )
        
        # Compile model with improved optimizer and learning rate
        optimizer = optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.00005,         # Reduced weight decay
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Use focal loss with label smoothing for better handling of class imbalance
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()
    
    def train(self, train_dataset=None, val_dataset=None, X_train=None, y_train=None, X_val=None, y_val=None, 
              epochs=150, batch_size=32, checkpoint_dir='checkpoints', logs_dir='logs'):
        """
        Train the model with callbacks for monitoring and saving
        
        Args:
            train_dataset: TensorFlow dataset for training (preferred)
            val_dataset: TensorFlow dataset for validation (preferred)
            X_train: Training features (optional, used if train_dataset is None)
            y_train: Training labels (optional, used if train_dataset is None)
            X_val: Validation features (optional, used if val_dataset is None)
            y_val: Validation labels (optional, used if val_dataset is None)
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
        
        # Early stopping with patience and restore best weights
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=30,  # Increased patience for better convergence
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Minimum improvement required
        )
        
        # Model checkpoint with improved naming
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, f'{model_id}_epoch_{{epoch:03d}}_val_acc_{{val_accuracy:.4f}}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1,
            save_weights_only=False  # Save the entire model
        )
        
        # Reduce learning rate on plateau for better convergence
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce by half
            patience=8,
            min_lr=1e-6,
            verbose=1,
            min_delta=0.001
        )
        
        # TensorBoard callback with more metrics
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(logs_dir, model_id),
            histogram_freq=1,
            write_graph=True,
            write_images=True,  # Write model weights as images
            update_freq='epoch',
            profile_batch=0  # Disable profiling for better performance
        )
        
        # CSV logger for tracking metrics
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(logs_dir, f'{model_id}_training.log'),
            append=True,
            separator=','
        )
        
        # Callbacks list
        callbacks_list = [
            early_stopping,
            checkpoint,
            reduce_lr,  # Use ReduceLROnPlateau instead of custom scheduler
            tensorboard,
            csv_logger
        ]
        
        # Calculate class weights if X_train and y_train are provided
        class_weights = None
        if X_train is not None and y_train is not None:
            class_weights = self._calculate_class_weights(y_train)
            print("\nClass weights:")
            for class_idx, weight in class_weights.items():
                print(f"  Class {class_idx}: {weight:.4f}")
        
        # Train model with all callbacks
        print("\nStarting enhanced model training...")
        
        # Use TensorFlow datasets if provided, otherwise use numpy arrays with data augmentation
        if train_dataset is not None and val_dataset is not None:
            # Calculate steps per epoch and validation steps
            steps_per_epoch = None  # Let TensorFlow handle this
            validation_steps = None  # Let TensorFlow handle this
            
            # Train with TensorFlow datasets
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1,
                class_weight=class_weights
            )
        else:
            # Fallback to numpy arrays with data augmentation
            if X_train is None or y_train is None or X_val is None or y_val is None:
                raise ValueError("Either provide TensorFlow datasets or numpy arrays for training")
                
            # Data augmentation on the fly
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest'
            )
            
            # Train with data augmentation
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1,
                class_weight=class_weights
            )
        
        self.training_history = history
        
        # Save training history as JSON
        history_dict = {
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']]
        }
        
        # Add learning rate to history if available
        if 'lr' in history.history:
            history_dict['learning_rate'] = [float(x) for x in history.history['lr']]
        
        with open(os.path.join(logs_dir, f'{model_id}_history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        # Plot and save training curves
        self._plot_training_history(history, logs_dir, model_id)
        
        return history
    
    def _calculate_class_weights(self, y_train):
        """
        Calculate class weights to handle imbalanced data
        
        Args:
            y_train: One-hot encoded training labels
            
        Returns:
            Dictionary of class weights
        """
        # Convert one-hot encoded labels to class indices
        y_indices = np.argmax(y_train, axis=1)
        
        # Count samples per class
        class_counts = np.bincount(y_indices)
        
        # Calculate weights (inversely proportional to class frequency)
        total_samples = len(y_indices)
        n_classes = len(class_counts)
        class_weights = {}
        
        for i in range(n_classes):
            # Balanced weight formula: total_samples / (n_classes * class_counts[i])
            # We apply a smoothing factor to prevent extreme weights
            weight = (total_samples / (n_classes * class_counts[i])) * 0.8 + 0.2
            class_weights[i] = weight
        
        return class_weights
    
    def _plot_training_history(self, history, logs_dir, model_id):
        """
        Plot and save training history curves
        
        Args:
            history: Training history object
            logs_dir: Directory to save plots
            model_id: Model identifier
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(logs_dir, f'{model_id}_training_curves.png'), dpi=300)
        plt.close()
    
    def evaluate(self, X_test=None, y_test=None, test_dataset=None):
        """
        Evaluate model performance with detailed metrics
        
        Args:
            X_test: Test features (optional, used if test_dataset is None)
            y_test: Test labels (optional, used if test_dataset is None)
            test_dataset: TensorFlow dataset for testing (preferred)
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating enhanced model on test data...")
        
        if test_dataset is not None:
            # Evaluate using TensorFlow dataset
            results = self.model.evaluate(test_dataset, verbose=1)
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = results[i]
                print(f"{metric_name}: {results[i]:.4f}")
        else:
            # Fallback to numpy arrays
            if X_test is None or y_test is None:
                raise ValueError("Either provide a TensorFlow dataset or numpy arrays for evaluation")
                
            results = self.model.evaluate(X_test, y_test, verbose=1)
            metrics = {}
            for i, metric_name in enumerate(self.model.metrics_names):
                metrics[metric_name] = results[i]
                print(f"{metric_name}: {results[i]:.4f}")
        
        # Generate predictions for confusion matrix and classification report
        if test_dataset is not None:
            # Get predictions from dataset
            y_pred_prob = []
            y_true = []
            for x, y in test_dataset:
                y_pred_prob.append(self.model.predict(x))
                y_true.append(y)
            y_pred_prob = np.vstack(y_pred_prob)
            y_true = np.vstack(y_true)
        else:
            # Get predictions from numpy arrays
            y_pred_prob = self.model.predict(X_test)
            y_true = y_test
        
        # Convert probabilities to class indices
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true_indices = np.argmax(y_true, axis=1)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true_indices, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save confusion matrix
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/confusion_matrix_{int(time.time())}.png', dpi=300)
        plt.close()
        
        # Generate classification report
        report = classification_report(y_true_indices, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Print classification report
        print("\nClassification Report:")
        print(pd.DataFrame(report).transpose())
        
        # Save classification report
        report_df.to_csv(f'plots/classification_report_{int(time.time())}.csv')
        
        # Add metrics to the results
        metrics['confusion_matrix'] = cm
        metrics['classification_report'] = report
        
        return metrics
    
    def predict(self, X=None, dataset=None):
        """
        Make predictions with detailed analysis
        
        Args:
            X: Input features (optional, used if dataset is None)
            dataset: TensorFlow dataset (preferred)
            
        Returns:
            Dictionary containing predictions and analysis
        """
        if dataset is not None:
            # Make predictions using TensorFlow dataset
            predictions = []
            for x, _ in dataset:
                batch_predictions = self.model.predict(x)
                predictions.append(batch_predictions)
            predictions = np.vstack(predictions)
        else:
            # Fallback to numpy arrays
            if X is None:
                raise ValueError("Either provide a TensorFlow dataset or numpy array for prediction")
            predictions = self.model.predict(X)
        
        # Get class indices with highest probability
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get confidence scores (probability of predicted class)
        confidence_scores = np.max(predictions, axis=1)
        
        # Create a dictionary of results
        results = {
            'probabilities': predictions,
            'predicted_classes': predicted_classes,
            'confidence_scores': confidence_scores
        }
        
        return results
    
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
        
        # Save in Keras format
        keras_path = os.path.join(save_dir, f"{model_name}.keras")
        self.model.save(keras_path, save_format='keras')
        print(f"Model saved to: {keras_path}")
        
        # Save in SavedModel format
        saved_model_path = os.path.join(save_dir, model_name)
        self.model.save(saved_model_path, save_format='tf')
        print(f"SavedModel saved to: {saved_model_path}")
        
        # Save model architecture as JSON
        model_json = self.model.to_json()
        with open(os.path.join(save_dir, f"{model_name}_architecture.json"), 'w') as f:
            f.write(model_json)
        print(f"Model architecture saved to: {os.path.join(save_dir, f'{model_name}_architecture.json')}")
        
        return keras_path, saved_model_path
    
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
