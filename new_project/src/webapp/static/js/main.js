/**
 * RAVDESS Emotion Recognition - Main JavaScript
 * Handles UI interactions, tab switching, and general functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize variables
    let socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);
    let emotionChart = null;
    
    // Initialize tabs
    initTabs();
    
    // Initialize emotion chart
    initEmotionChart();
    
    // Initialize event listeners
    initEventListeners();
    
    // Socket.io event handlers
    initSocketHandlers();
    
    /**
     * Initialize tab functionality
     */
    function initTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                // Remove active class from all buttons and contents
                tabBtns.forEach(b => b.classList.remove('active'));
                tabContents.forEach(c => c.classList.remove('active'));
                
                // Add active class to clicked button and corresponding content
                btn.classList.add('active');
                const tabId = btn.getAttribute('data-tab');
                document.getElementById(`${tabId}-tab`).classList.add('active');
            });
        });
    }
    
    /**
     * Initialize event listeners
     */
    function initEventListeners() {
        // Upload tab
        const uploadBtn = document.getElementById('uploadBtn');
        const audioFile = document.getElementById('audioFile');
        const dropArea = document.getElementById('dropArea');
        
        if (uploadBtn && audioFile) {
            uploadBtn.addEventListener('click', () => {
                audioFile.click();
            });
            
            audioFile.addEventListener('change', handleFileSelect);
        }
        
        if (dropArea) {
            // Drag and drop functionality
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => {
                    dropArea.classList.add('dragover');
                }, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, () => {
                    dropArea.classList.remove('dragover');
                }, false);
            });
            
            dropArea.addEventListener('drop', handleDrop, false);
        }
        
        // Live tab
        const startLiveBtn = document.getElementById('startLiveBtn');
        const stopLiveBtn = document.getElementById('stopLiveBtn');
        
        if (startLiveBtn && stopLiveBtn) {
            startLiveBtn.addEventListener('click', startLiveDetection);
            stopLiveBtn.addEventListener('click', stopLiveDetection);
        }
    }
    
    /**
     * Initialize Socket.io event handlers
     */
    function initSocketHandlers() {
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('emotion_result', (data) => {
            console.log('Received emotion result:', data);
            updateEmotionDisplay(data.emotion, data.confidence);
            updateEmotionChart(data.emotion, data.confidence);
        });
    }
    
    /**
     * Initialize emotion chart
     */
    function initEmotionChart() {
        const ctx = document.getElementById('emotionChart').getContext('2d');
        
        // Get available emotions from API
        fetch('/api/emotions')
            .then(response => response.json())
            .then(data => {
                const emotions = data.emotions || ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'];
                
                // Create empty data
                const chartData = emotions.map(emotion => ({
                    emotion: emotion,
                    confidence: 0
                }));
                
                // Create chart
                emotionChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: chartData.map(item => item.emotion),
                        datasets: [{
                            label: 'Confidence',
                            data: chartData.map(item => item.confidence),
                            backgroundColor: chartData.map(item => getEmotionColor(item.emotion, 0.7)),
                            borderColor: chartData.map(item => getEmotionColor(item.emotion, 1)),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                title: {
                                    display: true,
                                    text: 'Confidence'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Emotion'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Confidence: ${(context.raw * 100).toFixed(2)}%`;
                                    }
                                }
                            }
                        }
                    }
                });
            })
            .catch(error => {
                console.error('Error fetching emotions:', error);
                
                // Fallback to default emotions
                const emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'];
                
                // Create empty data
                const chartData = emotions.map(emotion => ({
                    emotion: emotion,
                    confidence: 0
                }));
                
                // Create chart with default emotions
                emotionChart = new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: chartData.map(item => item.emotion),
                        datasets: [{
                            label: 'Confidence',
                            data: chartData.map(item => item.confidence),
                            backgroundColor: chartData.map(item => getEmotionColor(item.emotion, 0.7)),
                            borderColor: chartData.map(item => getEmotionColor(item.emotion, 1)),
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 1,
                                title: {
                                    display: true,
                                    text: 'Confidence'
                                }
                            },
                            x: {
                                title: {
                                    display: true,
                                    text: 'Emotion'
                                }
                            }
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `Confidence: ${(context.raw * 100).toFixed(2)}%`;
                                    }
                                }
                            }
                        }
                    }
                });
            });
    }
    
    /**
     * Update emotion chart with new data
     */
    function updateEmotionChart(emotion, confidence) {
        if (!emotionChart) return;
        
        // Convert confidence to number if it's a string
        if (typeof confidence === 'string') {
            confidence = parseFloat(confidence) / 100;
        }
        
        // Find index of emotion
        const index = emotionChart.data.labels.findIndex(label => 
            label.toLowerCase() === emotion.toLowerCase());
        
        if (index !== -1) {
            // Reset all values to low
            emotionChart.data.datasets[0].data = emotionChart.data.datasets[0].data.map(() => 0.05);
            
            // Set the detected emotion's confidence
            emotionChart.data.datasets[0].data[index] = confidence;
            
            // Update chart
            emotionChart.update();
        }
    }
    
    /**
     * Prevent default behavior for events
     */
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    /**
     * Handle file drop
     */
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    }
    
    /**
     * Handle file selection
     */
    function handleFileSelect(e) {
        const files = e.target.files;
        
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    }
    
    /**
     * Handle file upload
     */
    function handleFileUpload(file) {
        // Check if file is audio
        if (!file.type.startsWith('audio/')) {
            alert('Please select an audio file');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds 10MB limit');
            return;
        }
        
        // Show upload status
        const uploadStatus = document.getElementById('uploadStatus');
        const statusText = uploadStatus.querySelector('.status-text');
        const progressBar = uploadStatus.querySelector('.progress');
        
        uploadStatus.classList.remove('hidden');
        statusText.textContent = 'Uploading...';
        progressBar.style.width = '50%';
        
        // Create form data
        const formData = new FormData();
        formData.append('audio', file);
        
        // Send file to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload successful:', data);
            
            // Update progress
            statusText.textContent = 'Processing...';
            progressBar.style.width = '100%';
            
            // Update emotion display
            updateEmotionDisplay(data.emotion, data.confidence);
            updateEmotionChart(data.emotion, data.confidence);
            
            // Hide upload status after a delay
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
                progressBar.style.width = '0%';
            }, 1000);
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            statusText.textContent = 'Error: ' + error.message;
            progressBar.style.width = '0%';
            
            // Hide upload status after a delay
            setTimeout(() => {
                uploadStatus.classList.add('hidden');
            }, 3000);
        });
    }
    
    /**
     * Start live emotion detection
     */
    function startLiveDetection() {
        // Update UI
        const startLiveBtn = document.getElementById('startLiveBtn');
        const stopLiveBtn = document.getElementById('stopLiveBtn');
        const liveStatus = document.getElementById('liveStatus');
        
        startLiveBtn.disabled = true;
        stopLiveBtn.disabled = false;
        liveStatus.classList.add('live');
        liveStatus.querySelector('.status-text').textContent = 'Listening...';
        
        // Start audio recording
        startAudioRecording(true);
    }
    
    /**
     * Stop live emotion detection
     */
    function stopLiveDetection() {
        // Update UI
        const startLiveBtn = document.getElementById('startLiveBtn');
        const stopLiveBtn = document.getElementById('stopLiveBtn');
        const liveStatus = document.getElementById('liveStatus');
        
        startLiveBtn.disabled = false;
        stopLiveBtn.disabled = true;
        liveStatus.classList.remove('live');
        liveStatus.querySelector('.status-text').textContent = 'Ready to start';
        
        // Stop audio recording
        stopAudioRecording();
    }
    
    /**
     * Update emotion display
     */
    function updateEmotionDisplay(emotion, confidence) {
        const emotionElement = document.querySelector('.emotion');
        const confidenceElement = document.querySelector('.confidence');
        const emotionIcon = document.querySelector('.emotion-icon i');
        
        // Update text
        emotionElement.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
        
        // Format confidence as percentage if it's a number
        if (typeof confidence === 'number') {
            confidenceElement.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;
        } else {
            confidenceElement.textContent = `Confidence: ${confidence}`;
        }
        
        // Update icon
        emotionIcon.className = getEmotionIcon(emotion);
        
        // Update color
        emotionElement.style.color = getEmotionColor(emotion);
    }
    
    /**
     * Get emotion color
     */
    function getEmotionColor(emotion, alpha = 1) {
        const colors = {
            'neutral': `rgba(144, 164, 174, ${alpha})`,
            'calm': `rgba(129, 199, 132, ${alpha})`,
            'happy': `rgba(255, 183, 77, ${alpha})`,
            'sad': `rgba(100, 181, 246, ${alpha})`,
            'angry': `rgba(229, 115, 115, ${alpha})`,
            'fearful': `rgba(149, 117, 205, ${alpha})`,
            'disgust': `rgba(161, 136, 127, ${alpha})`,
            'surprised': `rgba(77, 208, 225, ${alpha})`,
            'error': `rgba(244, 67, 54, ${alpha})`
        };
        
        return colors[emotion.toLowerCase()] || `rgba(158, 158, 158, ${alpha})`;
    }
    
    /**
     * Get emotion icon
     */
    function getEmotionIcon(emotion) {
        const icons = {
            'neutral': 'far fa-meh',
            'calm': 'far fa-smile',
            'happy': 'far fa-laugh',
            'sad': 'far fa-sad-tear',
            'angry': 'far fa-angry',
            'fearful': 'far fa-grimace',
            'disgust': 'far fa-dizzy',
            'surprised': 'far fa-surprise',
            'error': 'fas fa-exclamation-circle'
        };
        
        return icons[emotion.toLowerCase()] || 'far fa-question-circle';
    }
});
