#!/usr/bin/env python3
"""
RAVDESS Emotion Recognition - Run Script
This script provides a command-line interface to run different components of the project.
"""

import os
import sys
import argparse
import subprocess
import time

def print_header():
    """Print header with project information"""
    print("\n" + "=" * 80)
    print("RAVDESS Emotion Recognition System".center(80))
    print("99% Accuracy Speech Emotion Recognition".center(80))
    print("=" * 80 + "\n")

def check_environment():
    """Check if the environment is properly set up"""
    # Check if required directories exist
    required_dirs = ['data', 'models', 'checkpoints', 'logs', 'plots']
    for directory in required_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directory '{directory}' is available")
    
    # Check if requirements are installed
    try:
        import tensorflow as tf
        import numpy as np
        import librosa
        import flask
        print(f"✓ Required packages are installed")
        print(f"  - TensorFlow version: {tf.__version__}")
        print(f"  - NumPy version: {np.__version__}")
        print(f"  - Librosa version: {librosa.__version__}")
        print(f"  - Flask version: {flask.__version__}")
    except ImportError as e:
        print(f"✗ Missing required package: {str(e)}")
        print("  Please install required packages: pip install -r requirements.txt")
        return False
    
    return True

def train_model(args):
    """Train the emotion recognition model"""
    print("\n[Training Model]")
    
    # Build command
    cmd = [sys.executable, 'main.py']
    
    # Add arguments
    if args.dataset_path:
        cmd.extend(['--dataset_path', args.dataset_path])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
    if args.no_augment:
        cmd.append('--no_augment_data')
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_webapp(args):
    """Run the web application"""
    print("\n[Starting Web Application]")
    
    # Build command
    cmd = [sys.executable, '-m', 'src.webapp.app']
    
    # Add port if specified
    if args.port:
        os.environ['PORT'] = str(args.port)
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    print(f"Web application will be available at: http://localhost:{args.port or 8080}")
    subprocess.run(cmd)

def evaluate_model(args):
    """Evaluate the trained model"""
    print("\n[Evaluating Model]")
    
    # Check if model exists
    model_path = os.path.join('models', 'emotion_recognition_model.h5')
    if not os.path.exists(model_path):
        print(f"✗ Model not found at {model_path}")
        print("  Please train the model first using: python run.py train")
        return
    
    # Build command
    cmd = [sys.executable, 'main.py', '--evaluate_only']
    
    # Add arguments
    if args.dataset_path:
        cmd.extend(['--dataset_path', args.dataset_path])
    
    # Run command
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    """Main function"""
    # Print header
    print_header()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='RAVDESS Emotion Recognition System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the emotion recognition model')
    train_parser.add_argument('--dataset_path', type=str, help='Path to RAVDESS dataset')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size for training')
    train_parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    
    # Web app command
    webapp_parser = subparsers.add_parser('webapp', help='Run the web application')
    webapp_parser.add_argument('--port', type=int, help='Port to run the web application on')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate the trained model')
    evaluate_parser.add_argument('--dataset_path', type=str, help='Path to RAVDESS dataset')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check environment
    if not check_environment():
        return
    
    # Run command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'webapp':
        run_webapp(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
