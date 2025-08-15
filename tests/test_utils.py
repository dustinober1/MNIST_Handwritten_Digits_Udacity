"""Tests for utility functions."""

import pytest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.utils import (
    load_model, save_model, predict_digit, get_model_summary,
    setup_logging, get_device_info
)
from mnist_classifier.model import ImprovedMNISTNet, SimpleMNISTNet


class TestModelIO:
    """Test model saving and loading."""
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = ImprovedMNISTNet()
            save_path = str(Path(temp_dir) / "test_model.pth")
            
            # Save model
            success = save_model(model, save_path)
            assert success == True
            assert Path(save_path).exists()
            
            # Load model
            loaded_model = load_model(save_path)
            assert loaded_model is not None
            assert isinstance(loaded_model, ImprovedMNISTNet)
    
    def test_save_model_with_metadata(self):
        """Test saving model with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = SimpleMNISTNet()
            save_path = str(Path(temp_dir) / "test_model.pth")
            metadata = {
                "accuracy": 0.95,
                "epochs": 10,
                "batch_size": 64
            }
            
            # Save model with metadata
            success = save_model(model, save_path, metadata)
            assert success == True
            
            # Check metadata file exists
            metadata_path = str(Path(temp_dir) / "test_model.json")
            assert Path(metadata_path).exists()
            
            # Load and verify metadata
            with open(metadata_path, 'r') as f:
                loaded_metadata = json.load(f)
            
            assert loaded_metadata == metadata
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        result = load_model("nonexistent_model.pth")
        assert result is None
    
    def test_model_device_handling(self):
        """Test model loading with different devices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = ImprovedMNISTNet()
            save_path = str(Path(temp_dir) / "test_model.pth")
            
            # Save model
            save_model(model, save_path)
            
            # Load on CPU
            loaded_model_cpu = load_model(save_path, device="cpu")
            assert loaded_model_cpu is not None
            
            # Check device
            device = next(loaded_model_cpu.parameters()).device
            assert device.type == "cpu"
            
            # Load on CUDA if available
            if torch.cuda.is_available():
                loaded_model_gpu = load_model(save_path, device="cuda")
                assert loaded_model_gpu is not None
                
                device = next(loaded_model_gpu.parameters()).device
                assert device.type == "cuda"


class TestPrediction:
    """Test prediction functionality."""
    
    def test_predict_digit(self):
        """Test digit prediction."""
        model = ImprovedMNISTNet()
        model.eval()
        
        # Create test image
        test_image = Image.new('L', (28, 28), color=128)
        
        # Make prediction
        predicted_digit, confidence_scores, max_confidence = predict_digit(test_image, model)
        
        # Verify outputs
        assert isinstance(predicted_digit, int)
        assert 0 <= predicted_digit <= 9
        assert confidence_scores.shape == (10,)
        assert abs(confidence_scores.sum() - 1.0) < 1e-6  # Should sum to 1
        assert isinstance(max_confidence, float)
        assert 0.0 <= max_confidence <= 1.0
        assert max_confidence == confidence_scores[predicted_digit]
    
    def test_predict_digit_different_sizes(self):
        """Test prediction with different image sizes."""
        model = ImprovedMNISTNet()
        model.eval()
        
        sizes = [(10, 10), (28, 28), (50, 50), (100, 100)]
        
        for size in sizes:
            test_image = Image.new('L', size, color=100)
            predicted_digit, confidence_scores, max_confidence = predict_digit(test_image, model)
            
            assert 0 <= predicted_digit <= 9
            assert confidence_scores.shape == (10,)
    
    def test_predict_digit_invalid_input(self):
        """Test prediction with invalid input."""
        model = ImprovedMNISTNet()
        
        with pytest.raises(ValueError):
            predict_digit(None, model)


class TestModelSummary:
    """Test model summary functionality."""
    
    def test_get_model_summary_improved(self):
        """Test model summary for improved model."""
        model = ImprovedMNISTNet()
        summary = get_model_summary(model)
        
        assert isinstance(summary, dict)
        assert "model_class" in summary
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "model_size_mb" in summary
        
        assert summary["model_class"] == "ImprovedMNISTNet"
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] > 0
        assert summary["model_size_mb"] > 0
    
    def test_get_model_summary_simple(self):
        """Test model summary for simple model."""
        model = SimpleMNISTNet()
        summary = get_model_summary(model)
        
        assert summary["model_class"] == "SimpleMNISTNet"
        assert summary["total_parameters"] > 0
    
    def test_model_summary_comparison(self):
        """Test that improved model has more parameters than simple."""
        simple_model = SimpleMNISTNet()
        improved_model = ImprovedMNISTNet()
        
        simple_summary = get_model_summary(simple_model)
        improved_summary = get_model_summary(improved_model)
        
        assert improved_summary["total_parameters"] > simple_summary["total_parameters"]


class TestLogging:
    """Test logging functionality."""
    
    def test_setup_logging(self):
        """Test logging setup."""
        logger = setup_logging("INFO")
        assert logger is not None
        assert logger.name == "mnist_classifier"
        assert logger.level == 20  # INFO level
    
    def test_setup_logging_different_levels(self):
        """Test logging with different levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        expected_levels = [10, 20, 30, 40]
        
        for level, expected in zip(levels, expected_levels):
            logger = setup_logging(level)
            assert logger.level == expected


class TestDeviceInfo:
    """Test device information."""
    
    def test_get_device_info(self):
        """Test device info retrieval."""
        info = get_device_info()
        
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "device_count" in info
        assert isinstance(info["cuda_available"], bool)
        assert isinstance(info["device_count"], int)
        
        if info["cuda_available"]:
            assert "device_name" in info
            assert "device_properties" in info
            assert info["device_count"] > 0
        else:
            assert info["device_count"] == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_save_model_invalid_path(self):
        """Test saving model to invalid path."""
        model = ImprovedMNISTNet()
        
        # Try to save to a directory that doesn't exist and can't be created
        success = save_model(model, "/invalid/path/model.pth")
        assert success == False
    
    def test_predict_with_corrupted_model(self):
        """Test prediction with model in unusual state."""
        model = ImprovedMNISTNet()
        
        # Put model in training mode (should still work but might give different results)
        model.train()
        
        test_image = Image.new('L', (28, 28), color=128)
        predicted_digit, confidence_scores, max_confidence = predict_digit(test_image, model)
        
        assert 0 <= predicted_digit <= 9
        assert confidence_scores.shape == (10,)
    
    def test_model_consistency(self):
        """Test model gives consistent results for same input."""
        model = ImprovedMNISTNet()
        model.eval()
        
        test_image = Image.new('L', (28, 28), color=150)
        
        # Make multiple predictions
        predictions = []
        for _ in range(3):
            predicted_digit, _, _ = predict_digit(test_image, model)
            predictions.append(predicted_digit)
        
        # All predictions should be the same
        assert all(pred == predictions[0] for pred in predictions)