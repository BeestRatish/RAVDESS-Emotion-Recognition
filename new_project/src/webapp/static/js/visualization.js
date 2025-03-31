/**
 * RAVDESS Emotion Recognition - Visualization
 * Handles audio visualization using p5.js
 */

// Global variables
let p5Instance;
let mic;
let fft;
let amplitude;
let spectrum = [];
let waveform = [];
let volume = 0;
let isVisualizing = false;

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Create p5 instance
    const visualizationContainer = document.getElementById('visualization');
    
    if (visualizationContainer) {
        p5Instance = new p5(sketch, visualizationContainer);
    }
});

/**
 * p5.js sketch
 */
function sketch(p) {
    // Setup function
    p.setup = function() {
        // Create canvas to fill container
        const container = document.getElementById('visualization');
        const canvas = p.createCanvas(container.offsetWidth, container.offsetHeight);
        
        // Set up audio
        p.audioContext = getAudioContext();
        mic = new p5.AudioIn();
        fft = new p5.FFT(0.8, 1024);
        amplitude = new p5.Amplitude();
        
        // Connect mic to fft and amplitude
        mic.connect(fft);
        mic.connect(amplitude);
        
        // Set frame rate
        p.frameRate(30);
        
        // Set color mode
        p.colorMode(p.HSB, 360, 100, 100, 1);
        
        // Add event listeners for recording buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const startLiveBtn = document.getElementById('startLiveBtn');
        const stopLiveBtn = document.getElementById('stopLiveBtn');
        
        if (startBtn) {
            startBtn.addEventListener('click', startVisualization);
        }
        
        if (stopBtn) {
            stopBtn.addEventListener('click', stopVisualization);
        }
        
        if (startLiveBtn) {
            startLiveBtn.addEventListener('click', startVisualization);
        }
        
        if (stopLiveBtn) {
            stopLiveBtn.addEventListener('click', stopVisualization);
        }
    };
    
    // Draw function
    p.draw = function() {
        // Clear background
        p.background(240, 10, 98);
        
        // Only visualize if recording
        if (isVisualizing) {
            // Get audio data
            fft.analyze();
            spectrum = fft.analyze();
            waveform = fft.waveform();
            volume = amplitude.getLevel();
            
            // Draw visualization
            drawWaveform();
            drawSpectrum();
        } else {
            // Draw idle state
            drawIdleState();
        }
    };
    
    /**
     * Draw waveform visualization
     */
    function drawWaveform() {
        p.push();
        p.noFill();
        p.stroke(220, 70, 60, 0.8);
        p.strokeWeight(2);
        
        p.beginShape();
        for (let i = 0; i < waveform.length; i++) {
            const x = p.map(i, 0, waveform.length, 0, p.width);
            const y = p.map(waveform[i], -1, 1, p.height * 0.75, p.height * 0.25);
            p.vertex(x, y);
        }
        p.endShape();
        p.pop();
    }
    
    /**
     * Draw spectrum visualization
     */
    function drawSpectrum() {
        p.push();
        p.noStroke();
        
        // Only use lower half of spectrum for better visualization
        const spectrumLength = spectrum.length / 2;
        
        for (let i = 0; i < spectrumLength; i++) {
            // Map values
            const x = p.map(i, 0, spectrumLength, 0, p.width);
            const h = p.map(spectrum[i], 0, 255, 0, p.height * 0.25);
            
            // Calculate color based on frequency
            const hue = p.map(i, 0, spectrumLength, 220, 360);
            const saturation = 80;
            const brightness = 90;
            
            // Draw rectangle
            p.fill(hue, saturation, brightness, 0.7);
            p.rect(x, p.height * 0.25 - h, p.width / spectrumLength, h);
        }
        p.pop();
        
        // Draw volume indicator
        drawVolumeIndicator();
    }
    
    /**
     * Draw volume indicator
     */
    function drawVolumeIndicator() {
        p.push();
        const size = volume * 100;
        p.fill(220, 80, 90, 0.3);
        p.noStroke();
        p.ellipse(p.width / 2, p.height / 2, size, size);
        p.pop();
    }
    
    /**
     * Draw idle state
     */
    function drawIdleState() {
        p.push();
        
        // Draw text
        p.fill(220, 30, 60);
        p.textAlign(p.CENTER, p.CENTER);
        p.textSize(16);
        p.text('Start recording to see audio visualization', p.width / 2, p.height / 2);
        
        // Draw decorative elements
        p.noFill();
        p.stroke(220, 30, 60, 0.3);
        p.strokeWeight(1);
        
        // Draw sine wave
        p.beginShape();
        for (let i = 0; i < p.width; i += 5) {
            const y = p.height / 2 + Math.sin(i * 0.05) * 20;
            p.vertex(i, y);
        }
        p.endShape();
        
        p.pop();
    }
    
    /**
     * Window resize event
     */
    p.windowResized = function() {
        const container = document.getElementById('visualization');
        if (container) {
            p.resizeCanvas(container.offsetWidth, container.offsetHeight);
        }
    };
}

/**
 * Start audio visualization
 */
function startVisualization() {
    if (!mic) return;
    
    // Start microphone
    mic.start();
    isVisualizing = true;
}

/**
 * Stop audio visualization
 */
function stopVisualization() {
    if (!mic) return;
    
    // Stop microphone
    mic.stop();
    isVisualizing = false;
}
