# Deployment Guide for Jetson Orin Nano

This guide explains how to deploy the emotion recognition model on the Jetson Orin Nano.

## Prerequisites

1. Jetson Orin Nano with JetPack installed
2. Python 3.7+ installed
3. Required packages:
   ```bash
   pip3 install tensorflow tensorflow-gpu numpy librosa soundfile
   ```

## Deployment Steps

1. **Transfer Model Files**
   - Copy the following files from your training machine to the Jetson:
     - `models/emotion_model.tflite`
     - `models/label_classes.npy`
     - `data_preprocessing.py` (for feature extraction)

2. **Create Inference Script**
   Create a new file called `inference.py` on the Jetson:
   ```python
   import tensorflow as tf
   import numpy as np
   import librosa
   from data_preprocessing import AudioPreprocessor

   class EmotionPredictor:
       def __init__(self, model_path, label_classes_path):
           # Load TFLite model
           self.interpreter = tf.lite.Interpreter(model_path=model_path)
           self.interpreter.allocate_tensors()
           
           # Get input and output tensors
           self.input_details = self.interpreter.get_input_details()
           self.output_details = self.interpreter.get_output_details()
           
           # Load label classes
           self.label_classes = np.load(label_classes_path)
           
           # Initialize preprocessor
           self.preprocessor = AudioPreprocessor()
       
       def predict_emotion(self, audio_file_path):
           # Extract features
           features = self.preprocessor.extract_features(audio_file_path)
           if features is None:
               return None
           
           # Reshape features for model input
           features = features.reshape(1, features.shape[0], features.shape[1], 1)
           
           # Set input tensor
           self.interpreter.set_tensor(self.input_details[0]['index'], features)
           
           # Run inference
           self.interpreter.invoke()
           
           # Get prediction
           output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
           predicted_class = np.argmax(output_data[0])
           
           # Get emotion label
           emotion = self.label_classes[predicted_class]
           
           return emotion

   def main():
       # Initialize predictor
       predictor = EmotionPredictor(
           model_path='models/emotion_model.tflite',
           label_classes_path='models/label_classes.npy'
       )
       
       # Example usage
       audio_file = "path/to/test/audio.wav"
       emotion = predictor.predict_emotion(audio_file)
       if emotion:
           print(f"Predicted emotion: {emotion}")
       else:
           print("Error processing audio file")

   if __name__ == "__main__":
       main()
   ```

3. **Test Deployment**
   ```bash
   python3 inference.py
   ```

## Performance Optimization

1. **Enable TensorRT**
   ```python
   # Add to inference.py
   import tensorrt as trt
   
   # Convert TFLite to TensorRT
   converter = trt.TrtGraphConverterV2(
       input_saved_model_dir='models/emotion_model.tflite',
       precision_mode='FP16'
   )
   converter.convert()
   converter.save('models/emotion_model.trt')
   ```

2. **Batch Processing**
   - Modify the inference script to handle multiple audio files
   - Use TensorRT for faster inference

## Troubleshooting

1. **Memory Issues**
   - Reduce batch size
   - Use TensorRT for memory optimization
   - Monitor GPU memory usage with `tegrastats`

2. **Performance Issues**
   - Enable TensorRT
   - Use FP16 precision
   - Optimize input preprocessing

3. **Audio Processing Issues**
   - Check audio file format compatibility
   - Verify sample rate matches training data
   - Ensure proper audio preprocessing

## Additional Resources

- [Jetson Orin Nano Documentation](https://developer.nvidia.com/embedded/jetson-orin-nano)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite/guide) 