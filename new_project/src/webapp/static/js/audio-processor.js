/**
 * RAVDESS Emotion Recognition - Audio Processor
 * Handles audio recording, processing, and sending to server
 */

// Global variables
let audioContext;
let audioStream;
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let isLiveMode = false;
let recordingInterval;
let recordingDuration = 0;
let socket;

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize socket connection
    socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    
    // Initialize audio recording buttons
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    
    if (startBtn && stopBtn) {
        startBtn.addEventListener('click', function() {
            startAudioRecording(false);
        });
        
        stopBtn.addEventListener('click', function() {
            stopAudioRecording();
        });
    }
});

/**
 * Start audio recording
 * @param {boolean} liveMode - Whether to use live mode (continuous recording)
 */
function startAudioRecording(liveMode = false) {
    // Check if recording is already in progress
    if (isRecording) return;
    
    // Set mode
    isLiveMode = liveMode;
    
    // Reset variables
    audioChunks = [];
    recordingDuration = 0;
    
    // Update UI
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const recordingTimer = document.getElementById('recordingTimer');
    
    if (!isLiveMode && startBtn && stopBtn) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
    }
    
    if (!isLiveMode && recordingStatus) {
        recordingStatus.classList.add('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Recording...';
    }
    
    // Request microphone access
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(function(stream) {
            audioStream = stream;
            
            // Create audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            
            // Create media recorder
            mediaRecorder = new MediaRecorder(stream);
            
            // Handle data available event
            mediaRecorder.ondataavailable = function(event) {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
                
                // In live mode, process chunks as they come
                if (isLiveMode && audioChunks.length > 0) {
                    processAudioChunks();
                }
            };
            
            // Handle recording stop event
            mediaRecorder.onstop = function() {
                // Only process in non-live mode
                if (!isLiveMode) {
                    processAudioChunks();
                }
            };
            
            // Start recording
            mediaRecorder.start(isLiveMode ? 3000 : undefined); // In live mode, get data every 3 seconds
            isRecording = true;
            
            // Start timer
            if (!isLiveMode && recordingTimer) {
                recordingInterval = setInterval(updateRecordingTimer, 1000);
            }
            
            console.log('Recording started');
        })
        .catch(function(error) {
            console.error('Error accessing microphone:', error);
            alert('Error accessing microphone: ' + error.message);
            
            // Reset UI
            if (!isLiveMode && startBtn && stopBtn) {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
            
            if (!isLiveMode && recordingStatus) {
                recordingStatus.classList.remove('recording');
                recordingStatus.querySelector('.status-text').textContent = 'Error: ' + error.message;
            }
        });
}

/**
 * Stop audio recording
 */
function stopAudioRecording() {
    // Check if recording is in progress
    if (!isRecording || !mediaRecorder) return;
    
    // Stop media recorder
    mediaRecorder.stop();
    isRecording = false;
    
    // Stop audio tracks
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
    }
    
    // Clear timer
    if (recordingInterval) {
        clearInterval(recordingInterval);
        recordingInterval = null;
    }
    
    // Update UI
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    
    if (!isLiveMode && startBtn && stopBtn) {
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
    
    if (!isLiveMode && recordingStatus) {
        recordingStatus.classList.remove('recording');
        recordingStatus.querySelector('.status-text').textContent = 'Processing...';
    }
    
    console.log('Recording stopped');
}

/**
 * Process recorded audio chunks
 */
function processAudioChunks() {
    if (audioChunks.length === 0) return;
    
    // Create blob from chunks
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    
    // In live mode, clear chunks for next recording
    if (isLiveMode) {
        audioChunks = [];
    }
    
    // Convert blob to base64
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    
    reader.onloadend = function() {
        const base64data = reader.result;
        
        // Send to server via socket.io
        socket.emit('audio_data', base64data);
        
        // Update UI in non-live mode
        if (!isLiveMode) {
            const recordingStatus = document.getElementById('recordingStatus');
            if (recordingStatus) {
                recordingStatus.querySelector('.status-text').textContent = 'Ready to record';
            }
        }
    };
}

/**
 * Update recording timer
 */
function updateRecordingTimer() {
    recordingDuration++;
    
    const minutes = Math.floor(recordingDuration / 60);
    const seconds = recordingDuration % 60;
    
    const formattedTime = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    
    const recordingTimer = document.getElementById('recordingTimer');
    if (recordingTimer) {
        recordingTimer.textContent = formattedTime;
    }
    
    // Limit recording to 30 seconds in non-live mode
    if (!isLiveMode && recordingDuration >= 30) {
        stopAudioRecording();
    }
}
