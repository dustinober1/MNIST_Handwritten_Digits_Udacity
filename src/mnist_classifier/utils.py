"""Utility functions for MNIST digit classification."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import json
from PIL import Image
import numpy as np

from .model import ImprovedMNISTNet
from .preprocessing import preprocess_image, postprocess_prediction


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("mnist_classifier")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_model(model_path: str, device: Optional[str] = None) -> Optional[ImprovedMNISTNet]:
    """
    Load a trained MNIST model from file.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load model on ('cpu', 'cuda', or None for auto)
        
    Returns:
        Loaded model or None if loading fails
    """
    logger = setup_logging()
    
    try:
        if not Path(model_path).exists():
            logger.error(f"Model file not found at {model_path}")
            return None
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = ImprovedMNISTNet()
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        logger.info(f"Model loaded successfully from {model_path} on {device}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def save_model(model: nn.Module, save_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a trained model with optional metadata.
    
    Args:
        model: PyTorch model to save
        save_path: Path where to save the model
        metadata: Optional metadata dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Save model state dict
        torch.save(model.state_dict(), save_path)
        
        # Save metadata if provided
        if metadata:
            metadata_path = str(Path(save_path).with_suffix('.json'))
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved successfully to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        return False


def predict_digit(image: Image.Image, model: ImprovedMNISTNet) -> Tuple[int, np.ndarray, float]:
    """
    Predict digit from image using trained model.
    
    Args:
        image: PIL Image containing handwritten digit
        model: Trained MNIST model
        
    Returns:
        Tuple of (predicted_digit, confidence_scores, max_confidence)
        
    Raises:
        ValueError: If prediction fails
    """
    try:
        # Preprocess image
        input_tensor = preprocess_image(image, train=False)
        
        # Get device from model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
        
        # Postprocess results
        predicted_digit, confidence_scores, max_confidence = postprocess_prediction(output)
        
        return predicted_digit, confidence_scores, max_confidence
        
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed summary of model architecture and parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        "model_class": model.__class__.__name__,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    # Add architecture info if available
    if hasattr(model, 'get_model_info'):
        summary.update(model.get_model_info())
    
    return summary


def calculate_accuracy(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
    """
    Calculate model accuracy on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        device: Device to run calculations on
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total


def get_device_info() -> Dict[str, Any]:
    """
    Get information about available computing devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    }
    
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name()
        info["device_properties"] = str(torch.cuda.get_device_properties(0))
    
    return info