import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
import os
import time

class ModelEvaluator:
    """
    Class for evaluating the emotion recognition model.
    Provides comprehensive evaluation metrics and analysis.
    """
    def __init__(self, model, label_encoder, output_dir='evaluation'):
        """
        Initialize the evaluator
        
        Args:
            model: Trained model
            label_encoder: Label encoder for emotion classes
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.label_encoder = label_encoder
        self.output_dir = output_dir
        self.metrics = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_metrics(self, y_true, y_pred):
        """
        Generate comprehensive evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_micro = precision_score(y_true, y_pred, average='micro')
        precision_macro = precision_score(y_true, y_pred, average='macro')
        precision_weighted = precision_score(y_true, y_pred, average='weighted')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        recall_weighted = recall_score(y_true, y_pred, average='weighted')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        class_names = self.label_encoder.classes_
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': {
                'micro': float(precision_micro),
                'macro': float(precision_macro),
                'weighted': float(precision_weighted),
                'per_class': {class_name: float(precision) for class_name, precision in zip(class_names, precision_per_class)}
            },
            'recall': {
                'micro': float(recall_micro),
                'macro': float(recall_macro),
                'weighted': float(recall_weighted),
                'per_class': {class_name: float(recall) for class_name, recall in zip(class_names, recall_per_class)}
            },
            'f1': {
                'micro': float(f1_micro),
                'macro': float(f1_macro),
                'weighted': float(f1_weighted),
                'per_class': {class_name: float(f1) for class_name, f1 in zip(class_names, f1_per_class)}
            },
            'confusion_matrix': cm.tolist(),
            'class_names': class_names.tolist()
        }
        
        # Print summary
        self._print_metrics_summary()
        
        # Save metrics
        self._save_metrics()
        
        return self.metrics
    
    def _print_metrics_summary(self):
        """Print summary of evaluation metrics"""
        print("\n=== Evaluation Metrics ===")
        print(f"Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Precision (weighted): {self.metrics['precision']['weighted']:.4f}")
        print(f"Recall (weighted): {self.metrics['recall']['weighted']:.4f}")
        print(f"F1 Score (weighted): {self.metrics['f1']['weighted']:.4f}")
        
        print("\nPer-class F1 Scores:")
        for class_name, f1 in self.metrics['f1']['per_class'].items():
            print(f"  {class_name}: {f1:.4f}")
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        timestamp = int(time.time())
        metrics_path = os.path.join(self.output_dir, f'metrics_{timestamp}.json')
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nEvaluation metrics saved to: {metrics_path}")
    
    def generate_classification_report(self, y_true, y_pred):
        """
        Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as a dictionary
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        
        return report
    
    def evaluate_on_test_data(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Test loss and accuracy
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        return test_loss, test_accuracy
    
    def analyze_misclassifications(self, X_test, y_true, file_paths=None):
        """
        Analyze misclassified samples
        
        Args:
            X_test: Test features
            y_true: True labels
            file_paths: List of file paths corresponding to test samples
            
        Returns:
            DataFrame with misclassification analysis
        """
        # Generate predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert encoded labels to class names
        y_true_classes = self.label_encoder.inverse_transform(y_true)
        y_pred_classes = self.label_encoder.inverse_transform(y_pred)
        
        # Find misclassified samples
        misclassified_indices = np.where(y_true != y_pred)[0]
        
        # Create analysis dataframe
        data = []
        
        for idx in misclassified_indices:
            sample = {
                'index': idx,
                'true_label': y_true_classes[idx],
                'predicted_label': y_pred_classes[idx],
                'confidence': float(y_pred_proba[idx, y_pred[idx]]),
            }
            
            # Add file path if available
            if file_paths is not None:
                sample['file_path'] = file_paths[idx]
            
            data.append(sample)
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Sort by confidence
        df = df.sort_values('confidence', ascending=False)
        
        # Save to CSV
        timestamp = int(time.time())
        csv_path = os.path.join(self.output_dir, f'misclassifications_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"\nMisclassification analysis saved to: {csv_path}")
        print(f"Number of misclassified samples: {len(df)} out of {len(y_true)} ({len(df)/len(y_true)*100:.2f}%)")
        
        return df
    
    def evaluate_model_robustness(self, X_test, y_test, noise_levels=[0.01, 0.05, 0.1]):
        """
        Evaluate model robustness by adding different levels of noise
        
        Args:
            X_test: Test features
            y_test: Test labels (one-hot encoded)
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary with robustness evaluation results
        """
        robustness_results = {
            'noise_levels': noise_levels,
            'accuracy': []
        }
        
        print("\n=== Robustness Evaluation ===")
        print("Testing model performance with different noise levels:")
        
        # Evaluate on original data
        _, base_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Original accuracy (no noise): {base_accuracy:.4f}")
        
        # Evaluate with different noise levels
        for noise_level in noise_levels:
            # Add noise to test data
            X_test_noisy = X_test.copy()
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_test_noisy += noise
            
            # Evaluate
            _, accuracy = self.model.evaluate(X_test_noisy, y_test, verbose=0)
            robustness_results['accuracy'].append(float(accuracy))
            
            print(f"Accuracy with noise level {noise_level}: {accuracy:.4f}")
        
        # Save results
        timestamp = int(time.time())
        json_path = os.path.join(self.output_dir, f'robustness_{timestamp}.json')
        
        with open(json_path, 'w') as f:
            json.dump(robustness_results, f, indent=2)
        
        print(f"\nRobustness evaluation saved to: {json_path}")
        
        return robustness_results
