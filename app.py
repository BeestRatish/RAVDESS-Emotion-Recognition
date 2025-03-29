from flask import Flask, render_template
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

# Load the model and label encoder
model = tf.keras.models.load_model('models/final_model.h5')
label_encoder = np.load('models/label_classes.npy')

def process_audio(audio_data):
    """Process audio data and return emotion prediction"""
    try:
        # Convert audio data to numpy array
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=22050, duration=3)
        
        # Extract features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_chroma=40)
        
        # Ensure fixed length
        target_length = 180
        mfcc = librosa.util.fix_length(mfcc, size=target_length)
        mel_spec = librosa.util.fix_length(mel_spec, size=target_length)
        chroma = librosa.util.fix_length(chroma, size=target_length)
        
        # Combine features
        features = np.concatenate([mfcc, mel_spec, chroma], axis=0)
        
        # Reshape for model input
        features = features.reshape(1, features.shape[0], features.shape[1], 1)
        
        # Make prediction
        prediction = model.predict(features)
        emotion_idx = np.argmax(prediction[0])
        emotion = label_encoder[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        return emotion, confidence
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None, 0.0

@app.route('/')
def index():
    return render_template('index.html')

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
        print(f"Error handling audio data: {str(e)}")
        socketio.emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 