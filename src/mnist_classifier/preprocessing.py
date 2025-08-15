"""Image preprocessing utilities for MNIST digit classification."""

import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Tuple, Optional
import numpy as np


def get_mnist_transforms(train: bool = True) -> transforms.Compose:
    """
    Get standard MNIST preprocessing transforms.
    
    Args:
        train: If True, includes data augmentation transforms
        
    Returns:
        Composed transforms for MNIST preprocessing
    """
    base_transforms = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ]
    
    if train:
        # Add data augmentation for training
        augmentation_transforms = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ]
        # Insert augmentations before tensor conversion
        transforms_list = base_transforms[:2] + augmentation_transforms + base_transforms[2:]
        return transforms.Compose(transforms_list)
    
    return transforms.Compose(base_transforms)


def preprocess_image(image: Image.Image, train: bool = False) -> torch.Tensor:
    """
    Preprocess a PIL Image for MNIST model inference.
    
    Args:
        image: PIL Image to preprocess
        train: Whether to apply training augmentations
        
    Returns:
        Preprocessed tensor ready for model input
        
    Raises:
        ValueError: If image preprocessing fails
    """
    try:
        transform = get_mnist_transforms(train=train)
        tensor = transform(image)
        
        # Add batch dimension if not present
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            
        return tensor
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")


def postprocess_prediction(output: torch.Tensor) -> Tuple[int, np.ndarray, float]:
    """
    Postprocess model output to get prediction and confidence scores.
    
    Args:
        output: Raw model output tensor
        
    Returns:
        Tuple of (predicted_class, confidence_scores, max_confidence)
    """
    with torch.no_grad():
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence_scores = probabilities.squeeze().numpy()
        max_confidence = float(confidence_scores[predicted_class])
    
    return predicted_class, confidence_scores, max_confidence


def validate_image(image: Image.Image) -> bool:
    """
    Validate that an image is suitable for MNIST classification.
    
    Args:
        image: PIL Image to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Check if image can be converted to grayscale
        if image.mode not in ['L', 'RGB', 'RGBA']:
            return False
        
        # Check minimum size
        if image.size[0] < 8 or image.size[1] < 8:
            return False
        
        # Check maximum size (reasonable limit)
        if image.size[0] > 1000 or image.size[1] > 1000:
            return False
        
        return True
    except Exception:
        return False


def create_sample_digit_image(digit: int, size: Tuple[int, int] = (28, 28)) -> Image.Image:
    """
    Create a simple sample digit image for testing.
    
    Args:
        digit: Digit to create (0-9)
        size: Image size (width, height)
        
    Returns:
        PIL Image with simple digit representation
    """
    import numpy as np
    
    if not 0 <= digit <= 9:
        raise ValueError("Digit must be between 0 and 9")
    
    # Create a simple representation (this is just for testing)
    img_array = np.zeros(size, dtype=np.uint8)
    
    # Add some simple patterns for each digit (very basic)
    center_x, center_y = size[0] // 2, size[1] // 2
    
    if digit == 0:
        # Draw a circle-like pattern
        for i in range(size[0]):
            for j in range(size[1]):
                dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                if 8 <= dist <= 12:
                    img_array[j, i] = 255
    elif digit == 1:
        # Draw a vertical line
        img_array[:, center_x-1:center_x+2] = 255
    
    # For other digits, just create a simple pattern
    # (In a real application, you'd use proper digit generation)
    
    return Image.fromarray(img_array, mode='L')