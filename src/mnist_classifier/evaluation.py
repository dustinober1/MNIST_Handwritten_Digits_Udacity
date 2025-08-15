"""Model evaluation utilities and metrics for MNIST classifier."""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
from pathlib import Path


def evaluate_model(
    model: nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of the model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader for the dataset
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Get probabilities and predictions
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": class_report,
        "predictions": y_pred,
        "true_labels": y_true,
        "probabilities": y_prob
    }


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, 
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix with proper formatting.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names (default: 0-9 for MNIST)
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    if class_names is None:
        class_names = [str(i) for i in range(confusion_matrix.shape[0])]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate percentages
    cm_percent = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(
        cm_percent, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_class_performance(
    classification_report: Dict[str, Any],
    title: str = "Per-Class Performance",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        classification_report: Classification report from sklearn
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Extract per-class metrics (excluding average metrics)
    classes = [k for k in classification_report.keys() 
               if k.isdigit() or (isinstance(k, str) and k.isdigit())]
    
    metrics_data = []
    for class_name in classes:
        class_metrics = classification_report[class_name]
        metrics_data.append({
            'Class': class_name,
            'Precision': class_metrics['precision'],
            'Recall': class_metrics['recall'],
            'F1-Score': class_metrics['f1-score']
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_misclassifications(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Analyze the most common misclassification patterns.
    
    Args:
        true_labels: Array of true labels
        predictions: Array of predicted labels
        probabilities: Array of prediction probabilities
        top_n: Number of top misclassification pairs to return
        
    Returns:
        Dictionary with misclassification analysis
    """
    # Find misclassified samples
    misclassified_mask = true_labels != predictions
    misclassified_true = true_labels[misclassified_mask]
    misclassified_pred = predictions[misclassified_mask]
    misclassified_prob = probabilities[misclassified_mask]
    
    # Count misclassification pairs
    from collections import Counter
    misclass_pairs = [(true, pred) for true, pred in zip(misclassified_true, misclassified_pred)]
    misclass_counts = Counter(misclass_pairs)
    
    # Get confidence scores for misclassified samples
    misclass_confidences = [prob[pred] for prob, pred in zip(misclassified_prob, misclassified_pred)]
    
    return {
        "total_misclassified": len(misclassified_true),
        "misclassification_rate": len(misclassified_true) / len(true_labels),
        "top_misclassification_pairs": misclass_counts.most_common(top_n),
        "avg_misclassification_confidence": np.mean(misclass_confidences),
        "low_confidence_threshold": np.percentile(misclass_confidences, 25)
    }


def plot_prediction_confidence_distribution(
    probabilities: np.ndarray,
    predictions: np.ndarray,
    true_labels: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot distribution of prediction confidences for correct vs incorrect predictions.
    
    Args:
        probabilities: Array of prediction probabilities
        predictions: Array of predicted labels
        true_labels: Array of true labels
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Get confidence scores (max probability for each prediction)
    confidence_scores = np.max(probabilities, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = predictions == true_labels
    correct_confidences = confidence_scores[correct_mask]
    incorrect_confidences = confidence_scores[~correct_mask]
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', density=True)
    ax1.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', density=True)
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Density')
    ax1.set_title('Confidence Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    data_to_plot = [correct_confidences, incorrect_confidences]
    ax2.boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
    ax2.set_ylabel('Prediction Confidence')
    ax2.set_title('Confidence Box Plot')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def generate_evaluation_report(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with plots and metrics.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save plots and report
        
    Returns:
        Complete evaluation results
    """
    print("Starting model evaluation...")
    
    # Run evaluation
    results = evaluate_model(model, test_loader, device)
    
    # Create save directory if specified
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if save_dir:
        # Confusion matrix
        cm_path = save_path / "confusion_matrix.png"
        plot_confusion_matrix(results["confusion_matrix"], save_path=str(cm_path))
        
        # Class performance
        perf_path = save_path / "class_performance.png"
        plot_class_performance(results["classification_report"], save_path=str(perf_path))
        
        # Confidence distribution
        conf_path = save_path / "confidence_distribution.png"
        plot_prediction_confidence_distribution(
            results["probabilities"],
            results["predictions"],
            results["true_labels"],
            save_path=str(conf_path)
        )
    
    # Analyze misclassifications
    misclass_analysis = analyze_misclassifications(
        results["true_labels"],
        results["predictions"],
        results["probabilities"]
    )
    
    results["misclassification_analysis"] = misclass_analysis
    
    print(f"Evaluation complete!")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Misclassification Rate: {misclass_analysis['misclassification_rate']:.4f}")
    
    return results