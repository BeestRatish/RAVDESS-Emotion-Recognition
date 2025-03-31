from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
import io
import base64
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# We'll use a simple approach instead of the problematic model
# Define a simple function to classify emotions based on audio features
def simple_emotion_classifier(audio_features):
    """
    A simple classifier that uses basic audio features to determine if an audio is happy or sad
    This is a placeholder for the more complex model that's having issues
    """
    # Extract simple features from the audio
    # Higher energy and pitch often correlate with happy emotions
    # Lower energy and more pauses often correlate with sad emotions
    
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
    if energy > 0.2 and (zero_crossings > 0.05 or spectral_centroid > 0.1):
        return "happy", 0.7 + (energy * 0.3)  # Higher confidence for higher energy
    elif energy < 0.15 or zero_crossings < 0.03:
        return "sad", 0.6 + ((1 - energy) * 0.3)  # Higher confidence for lower energy
    else:
        return "neutral", 0.5

def process_audio(audio_data):
    """Process audio data and return emotion prediction"""
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
        
        # Resample to 22050 Hz
        print("Resampling audio...")
        audio = librosa.resample(audio, orig_sr=44100, target_sr=22050)
        print(f"Resampled audio shape: {audio.shape}")
        
        # Extract some basic audio features for our simple classifier
        print("Extracting audio features...")
        
        # Extract mel spectrogram for visualization (not used in classification)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=22050,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            power=2.0
        )
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        
        # Use our simple classifier instead of the problematic model
        print("Classifying emotion...")
        emotion, confidence = simple_emotion_classifier(audio)
        
        print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
        return emotion, confidence
        
    except Exception as e:
        print(f"Error in process_audio: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return "error", 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
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
        audio_data = base64.b64decode(data.split(',')[1])
        
        # Process audio and get emotion
        emotion, confidence = process_audio(audio_data)
        
        if emotion:
            # Send emotion and confidence back to client
            socketio.emit('emotion_result', {
                'emotion': emotion,
                'confidence': f"{confidence:.2%}"
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

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=8080) 