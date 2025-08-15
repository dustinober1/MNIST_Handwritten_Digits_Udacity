"""Pytest configuration and fixtures for MNIST classifier tests."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Fixture to provide device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_batch():
    """Fixture to provide sample batch data."""
    batch_size = 8
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


@pytest.fixture
def trained_model():
    """Fixture to provide a simple trained model for testing."""
    from mnist_classifier.model import ImprovedMNISTNet
    
    model = ImprovedMNISTNet()
    model.eval()
    return model