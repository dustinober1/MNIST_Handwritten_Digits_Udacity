"""Neural network model definitions for MNIST digit classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ImprovedMNISTNet(nn.Module):
    """
    Improved neural network for MNIST digit classification.
    
    Architecture:
    - Input: 28x28 grayscale images (flattened to 784 features)
    - Hidden layers: 256 -> 128 -> 64 neurons with ReLU activation
    - Output: 10 classes (digits 0-9)
    """
    
    def __init__(self, input_size: int = 28 * 28, hidden_sizes: Tuple[int, ...] = (256, 128, 64), num_classes: int = 10):
        """
        Initialize the improved MNIST network.
        
        Args:
            input_size: Size of flattened input (default: 784 for 28x28 images)
            hidden_sizes: Tuple of hidden layer sizes
            num_classes: Number of output classes (default: 10 for digits 0-9)
        """
        super(ImprovedMNISTNet, self).__init__()
        
        self.flatten = nn.Flatten()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        # Store layers
        self.fc1 = layers[0]
        self.fc2 = layers[1]
        self.fc3 = layers[2]
        self.fc4 = layers[3]
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for output layer
        return x
    
    def get_model_info(self) -> dict:
        """Get information about the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "architecture": "Fully Connected Neural Network",
            "layers": [
                f"Input: {28*28} features",
                f"Hidden 1: {self.fc1.out_features} neurons (ReLU)",
                f"Hidden 2: {self.fc2.out_features} neurons (ReLU)", 
                f"Hidden 3: {self.fc3.out_features} neurons (ReLU)",
                f"Output: {self.fc4.out_features} classes"
            ]
        }


class SimpleMNISTNet(nn.Module):
    """
    Simple baseline neural network for MNIST digit classification.
    
    Simpler architecture for comparison with the improved model.
    """
    
    def __init__(self):
        """Initialize the simple MNIST network."""
        super(SimpleMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the simple network."""
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x