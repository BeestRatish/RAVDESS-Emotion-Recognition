import os
import sys
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO

# Add parent directory to path to import from other modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from preprocessing.audio_processor import AudioProcessor

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'emotion_recognition_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/emotion_recognition_model.h5')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), '../../models/label_classes.npy')
SAMPLE_RATE = 22050
DURATION = 3

# Initialize audio processor
audio_processor = AudioProcessor(sample_rate=SAMPLE_RATE, duration=DURATION)

# Load model and label encoder if they exist
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if os.path.exists(LABEL_ENCODER_PATH):
    try:
        audio_processor.load_label_encoder(LABEL_ENCODER_PATH)
        print(f"Label encoder loaded from {LABEL_ENCODER_PATH}")
    except Exception as e:
        print(f"Error loading label encoder: {str(e)}")

def extract_features_for_prediction(audio_data):
    """
    Extract features from audio data for prediction
    
    Args:
        audio_data: Audio data as numpy array
        
    Returns:
        Extracted features ready for model input
    """
    try:
        # Resample if needed
        if len(audio_data) > SAMPLE_RATE * DURATION:
            audio_data = audio_data[:int(SAMPLE_RATE * DURATION)]
        elif len(audio_data) < SAMPLE_RATE * DURATION:
            # Pad with zeros if audio is too short
            padding = int(SAMPLE_RATE * DURATION) - len(audio_data)
            audio_data = np.pad(audio_data, (0, padding), 'constant')
        
        # Extract features (simplified version of AudioProcessor.extract_features)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=40)
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=40)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=SAMPLE_RATE, n_chroma=40)
        
        # Ensure all features have the same time dimension
        target_length = 180
        mfcc = librosa.util.fix_length(mfcc, size=target_length)
        mel_spec = librosa.util.fix_length(mel_spec, size=target_length)
        chroma = librosa.util.fix_length(chroma, size=target_length)
        
        # Additional features
        zcr = librosa.feature.zero_crossing_rate(audio_data)
        zcr = librosa.util.fix_length(zcr, size=target_length)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=SAMPLE_RATE)
        spectral_centroid = librosa.util.fix_length(spectral_centroid, size=target_length)
        
        rms = librosa.feature.rms(y=audio_data)
        rms = librosa.util.fix_length(rms, size=target_length)
        
        # Combine features
        features = np.concatenate([mfcc, mel_spec, chroma, zcr, spectral_centroid, rms], axis=0)
        
        # Reshape for model input (batch_size, height, width, channels)
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def process_audio(audio_data):
    """
    Process audio data and return emotion prediction
    
    Args:
        audio_data: Audio data as bytes or base64 string
        
    Returns:
        Predicted emotion and confidence
    """
    try:
        print("Starting audio processing...")
        
        # Convert audio data to numpy array
        if isinstance(audio_data, (str, bytes)):
            try:
                # Handle base64 encoded audio data
                if isinstance(audio_data, str) and ',' in audio_data:
                    print("Processing base64 encoded string...")
                    audio_base64 = audio_data.split(',')[1]
                    audio_data = base64.b64decode(audio_base64.encode('utf-8'))
                elif isinstance(audio_data, str):
                    print("Processing raw base64 string...")
                    audio_data = base64.b64decode(audio_data)
                print(f"Decoded audio data length: {len(audio_data)} bytes")
            except Exception as e:
                print(f"Error decoding base64: {str(e)}")
                raise
        
        # Convert to numpy array
        print("Converting to numpy array...")
        audio = np.frombuffer(audio_data, dtype=np.int16)
        if len(audio) == 0:
            raise ValueError("Empty audio data")
        print(f"Audio array shape: {audio.shape}, dtype: {audio.dtype}")
        
        # Convert to float32 and normalize
        audio = audio.astype(np.float32)
        audio = audio / np.max(np.abs(audio))  # Normalize to [-1, 1]
        print("Audio normalized successfully")
        
        # Resample to target sample rate
        print("Resampling audio...")
        audio = librosa.resample(audio, orig_sr=44100, target_sr=SAMPLE_RATE)
        print(f"Resampled audio shape: {audio.shape}")
        
        # Extract features for prediction
        print("Extracting audio features...")
        features = extract_features_for_prediction(audio)
        if features is None:
            return "error", 0.0
        
        # Make prediction if model is available
        if model is not None:
            print("Making prediction with model...")
            prediction = model.predict(features)
            predicted_class_index = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_index])
            
            # Convert class index to emotion name
            emotion = audio_processor.label_encoder.inverse_transform([predicted_class_index])[0]
            
            print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
            return emotion, confidence
        else:
            # Fallback to simple classifier if model is not available
            print("Model not available, using simple classifier...")
            return simple_emotion_classifier(audio)
        
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return "error", 0.0

def simple_emotion_classifier(audio_features):
    """
    A simple classifier that uses basic audio features to determine emotion
    This is a fallback for when the model is not available
    
    Args:
        audio_features: Audio data as numpy array
        
    Returns:
        Predicted emotion and confidence
    """
    # Calculate energy (volume)
    energy = np.mean(np.abs(audio_features))
    
    # Calculate zero-crossing rate (related to pitch)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio_features)))) / len(audio_features)
    
    # Calculate spectral centroid (brightness of sound)
    if len(audio_features) > 1:
        magnitudes = np.abs(np.fft.rfft(audio_features))
        freqs = np.fft.rfftfreq(len(audio_features))
        spectral_centroid = np.sum(magnitudes * freqs) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
    else:
        spectral_centroid = 0
    
    # Simple decision tree for classification
    if energy > 0.2 and zero_crossings > 0.05:
        if spectral_centroid > 0.1:
            return "happy", 0.7 + (energy * 0.3)
        else:
            return "angry", 0.65 + (energy * 0.3)
    elif energy > 0.15 and zero_crossings > 0.04:
        return "surprised", 0.6 + (energy * 0.2)
    elif energy < 0.1 and zero_crossings < 0.03:
        return "sad", 0.6 + ((1 - energy) * 0.3)
    elif energy < 0.15:
        return "fearful", 0.55 + ((1 - energy) * 0.2)
    else:
        return "neutral", 0.5

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the audio file
    audio_data = file.read()
    
    # Process the audio and get emotion
    emotion, confidence = process_audio(audio_data)
    
    return jsonify({
        'emotion': emotion,
        'confidence': float(confidence)
    })

@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data from the client"""
    try:
        # Decode base64 audio data
        audio_data = data.split(',')[1] if ',' in data else data
        audio_data = base64.b64decode(audio_data)
        
        # Process audio and get emotion
        emotion, confidence = process_audio(audio_data)
        
        if emotion:
            # Send emotion and confidence back to client
            socketio.emit('emotion_result', {
                'emotion': emotion,
                'confidence': float(confidence)
            })
            
    except Exception as e:
        print("\n=== Error in handle_audio_data ===")
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        socketio.emit('emotion_result', {
            'emotion': 'error',
            'confidence': 0.0,
            'error': str(e)
        })

@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """API endpoint to get available emotions"""
    if hasattr(audio_processor, 'label_encoder') and hasattr(audio_processor.label_encoder, 'classes_'):
        emotions = audio_processor.label_encoder.classes_.tolist()
    else:
        # Fallback if label encoder is not available
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    
    return jsonify({'emotions': emotions})

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """API endpoint to get model information"""
    model_info = {
        'model_available': model is not None,
        'model_path': MODEL_PATH if model is not None else None,
        'label_encoder_available': hasattr(audio_processor, 'label_encoder') and hasattr(audio_processor.label_encoder, 'classes_'),
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION
    }
    
    return jsonify(model_info)

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 8080))
    
    # Run the app
    print(f"Starting server on port {port}...")
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
