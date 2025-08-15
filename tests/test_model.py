"""Tests for MNIST model architecture."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.model import ImprovedMNISTNet, SimpleMNISTNet


class TestImprovedMNISTNet:
    """Test cases for ImprovedMNISTNet model."""
    
    def test_model_initialization(self):
        """Test model can be initialized correctly."""
        model = ImprovedMNISTNet()
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert hasattr(model, 'fc4')
    
    def test_model_forward_pass(self):
        """Test model forward pass with correct input shape."""
        model = ImprovedMNISTNet()
        batch_size = 8
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        
        output = model(input_tensor)
        
        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_model_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = ImprovedMNISTNet()
        
        for batch_size in [1, 16, 32, 64]:
            input_tensor = torch.randn(batch_size, 1, 28, 28)
            output = model(input_tensor)
            assert output.shape == (batch_size, 10)
    
    def test_model_gradients(self):
        """Test model gradients are computed correctly."""
        model = ImprovedMNISTNet()
        input_tensor = torch.randn(4, 1, 28, 28, requires_grad=True)
        target = torch.randint(0, 10, (4,))
        
        criterion = nn.CrossEntropyLoss()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()\n    \n    def test_model_evaluation_mode(self):\n        \"\"\"Test model behaves correctly in evaluation mode.\"\"\"\n        model = ImprovedMNISTNet()\n        input_tensor = torch.randn(4, 1, 28, 28)\n        \n        # Training mode\n        model.train()\n        output_train = model(input_tensor)\n        \n        # Evaluation mode\n        model.eval()\n        output_eval = model(input_tensor)\n        \n        # Outputs should be identical for this architecture\n        assert torch.allclose(output_train, output_eval)\n    \n    def test_model_custom_architecture(self):\n        \"\"\"Test model with custom architecture parameters.\"\"\"\n        custom_model = ImprovedMNISTNet(\n            input_size=784,\n            hidden_sizes=(128, 64),\n            num_classes=10\n        )\n        \n        input_tensor = torch.randn(4, 1, 28, 28)\n        output = custom_model(input_tensor)\n        \n        assert output.shape == (4, 10)\n    \n    def test_model_info(self):\n        \"\"\"Test model info method.\"\"\"\n        model = ImprovedMNISTNet()\n        info = model.get_model_info()\n        \n        assert isinstance(info, dict)\n        assert 'total_parameters' in info\n        assert 'trainable_parameters' in info\n        assert 'architecture' in info\n        assert 'layers' in info\n        assert info['total_parameters'] > 0\n\n\nclass TestSimpleMNISTNet:\n    \"\"\"Test cases for SimpleMNISTNet model.\"\"\"\n    \n    def test_simple_model_initialization(self):\n        \"\"\"Test simple model can be initialized correctly.\"\"\"\n        model = SimpleMNISTNet()\n        assert isinstance(model, nn.Module)\n        assert hasattr(model, 'fc1')\n        assert hasattr(model, 'fc2')\n        assert hasattr(model, 'fc3')\n    \n    def test_simple_model_forward_pass(self):\n        \"\"\"Test simple model forward pass.\"\"\"\n        model = SimpleMNISTNet()\n        batch_size = 8\n        input_tensor = torch.randn(batch_size, 1, 28, 28)\n        \n        output = model(input_tensor)\n        \n        assert output.shape == (batch_size, 10)\n        assert not torch.isnan(output).any()\n    \n    def test_model_comparison(self):\n        \"\"\"Test that improved model has more parameters than simple model.\"\"\"\n        simple_model = SimpleMNISTNet()\n        improved_model = ImprovedMNISTNet()\n        \n        simple_params = sum(p.numel() for p in simple_model.parameters())\n        improved_params = sum(p.numel() for p in improved_model.parameters())\n        \n        assert improved_params > simple_params\n\n\nclass TestModelEdgeCases:\n    \"\"\"Test edge cases and error conditions.\"\"\"\n    \n    def test_invalid_input_shape(self):\n        \"\"\"Test model behavior with invalid input shapes.\"\"\"\n        model = ImprovedMNISTNet()\n        \n        # Wrong number of dimensions\n        with pytest.raises(Exception):\n            invalid_input = torch.randn(8, 28, 28)  # Missing channel dimension\n            model(invalid_input)\n    \n    def test_empty_batch(self):\n        \"\"\"Test model with empty batch.\"\"\"\n        model = ImprovedMNISTNet()\n        empty_input = torch.randn(0, 1, 28, 28)\n        \n        output = model(empty_input)\n        assert output.shape == (0, 10)\n    \n    def test_model_device_consistency(self):\n        \"\"\"Test model works consistently across devices.\"\"\"\n        model = ImprovedMNISTNet()\n        input_tensor = torch.randn(4, 1, 28, 28)\n        \n        # CPU\n        model_cpu = model.to('cpu')\n        input_cpu = input_tensor.to('cpu')\n        output_cpu = model_cpu(input_cpu)\n        \n        assert output_cpu.device.type == 'cpu'\n        \n        # GPU (if available)\n        if torch.cuda.is_available():\n            model_gpu = model.to('cuda')\n            input_gpu = input_tensor.to('cuda')\n            output_gpu = model_gpu(input_gpu)\n            \n            assert output_gpu.device.type == 'cuda'\n            \n            # Results should be close (accounting for numerical differences)\n            assert torch.allclose(\n                output_cpu, \n                output_gpu.cpu(), \n                rtol=1e-5, \n                atol=1e-6\n            )"}