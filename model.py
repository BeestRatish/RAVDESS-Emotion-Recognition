import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
import os

class EmotionRecognitionModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
    
    def build_model(self):
        """Build improved CNN-LSTM model architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First CNN block with batch normalization
            layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            
            # Second CNN block with batch normalization
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Third CNN block with batch normalization
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),
            
            # Reshape for LSTM
            layers.Reshape((-1, 128)),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Bidirectional(layers.LSTM(64)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with improved optimizer and learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model with improved callbacks"""
        # Create checkpoints directory
        checkpoint_dir = 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Early stopping with patience
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        # Learning rate scheduler
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001
        )
        
        # CSV logger for tracking metrics
        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(checkpoint_dir, 'training.log'),
            append=True
        )
        
        # Progress bar callback
        progress_bar = tf.keras.callbacks.ProgbarLogger()
        
        # Train model with all callbacks
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                early_stopping,
                checkpoint,
                reduce_lr,
                csv_logger,
                progress_bar
            ],
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        self.model.load_weights(checkpoint_path)
        print(f"Loaded model from checkpoint: {checkpoint_path}") 