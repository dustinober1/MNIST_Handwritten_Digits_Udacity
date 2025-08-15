#!/usr/bin/env python3
"""Training script for MNIST digit classifier."""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.model import ImprovedMNISTNet, SimpleMNISTNet
from mnist_classifier.preprocessing import get_mnist_transforms
from mnist_classifier.utils import save_model, setup_logging, get_device_info
from mnist_classifier.evaluation import generate_evaluation_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MNIST digit classifier")
    
    parser.add_argument("--model", type=str, choices=["simple", "improved"], 
                       default="improved", help="Model architecture to use")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, 
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, 
                       help="Learning rate for optimizer")
    parser.add_argument("--data-dir", type=str, default="./data", 
                       help="Directory to store dataset")
    parser.add_argument("--output-dir", type=str, default="./models", 
                       help="Directory to save trained model")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], 
                       default="auto", help="Device to use for training")
    parser.add_argument("--augment", action="store_true", 
                       help="Use data augmentation")
    parser.add_argument("--evaluate", action="store_true", 
                       help="Run evaluation after training")
    parser.add_argument("--plot-results", action="store_true", 
                       help="Generate training plots")
    
    return parser.parse_args()


def create_data_loaders(data_dir: str, batch_size: int, augment: bool = False):
    """Create training and test data loaders."""
    # Training transforms
    train_transform = get_mnist_transforms(train=augment)
    test_transform = get_mnist_transforms(train=False)
    
    # Create datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, args, device):
    """Train the model and return training history."""
    logger = setup_logging()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            # Print progress
            if batch_idx % 200 == 0:
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], '
                          f'Batch [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')\n        \n        # Calculate epoch metrics\n        epoch_loss = running_loss / len(train_loader)\n        train_accuracy = correct_train / total_train\n        \n        # Evaluation phase\n        model.eval()\n        correct_test = 0\n        total_test = 0\n        \n        with torch.no_grad():\n            for images, labels in test_loader:\n                images, labels = images.to(device), labels.to(device)\n                outputs = model(images)\n                _, predicted = torch.max(outputs, 1)\n                total_test += labels.size(0)\n                correct_test += (predicted == labels).sum().item()\n        \n        test_accuracy = correct_test / total_test\n        \n        # Store metrics\n        train_losses.append(epoch_loss)\n        train_accuracies.append(train_accuracy)\n        test_accuracies.append(test_accuracy)\n        \n        logger.info(f'Epoch [{epoch+1}/{args.epochs}] - '\n                   f'Train Loss: {epoch_loss:.4f}, '\n                   f'Train Acc: {train_accuracy:.4f}, '\n                   f'Test Acc: {test_accuracy:.4f}')\n    \n    return {\n        'train_losses': train_losses,\n        'train_accuracies': train_accuracies,\n        'test_accuracies': test_accuracies\n    }\n\n\ndef plot_training_history(history, save_path=None):\n    \"\"\"Plot training history.\"\"\"\n    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))\n    \n    # Plot loss\n    ax1.plot(history['train_losses'], label='Training Loss')\n    ax1.set_title('Training Loss')\n    ax1.set_xlabel('Epoch')\n    ax1.set_ylabel('Loss')\n    ax1.legend()\n    ax1.grid(True)\n    \n    # Plot accuracy\n    ax2.plot(history['train_accuracies'], label='Training Accuracy')\n    ax2.plot(history['test_accuracies'], label='Test Accuracy')\n    ax2.set_title('Model Accuracy')\n    ax2.set_xlabel('Epoch')\n    ax2.set_ylabel('Accuracy')\n    ax2.legend()\n    ax2.grid(True)\n    \n    plt.tight_layout()\n    \n    if save_path:\n        fig.savefig(save_path, dpi=300, bbox_inches='tight')\n        print(f\"Training plots saved to {save_path}\")\n    \n    return fig\n\n\ndef main():\n    \"\"\"Main training function.\"\"\"\n    args = parse_arguments()\n    logger = setup_logging()\n    \n    # Setup device\n    if args.device == \"auto\":\n        device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n    else:\n        device = args.device\n    \n    logger.info(f\"Using device: {device}\")\n    \n    # Print device info\n    device_info = get_device_info()\n    logger.info(f\"Device info: {device_info}\")\n    \n    # Create output directory\n    output_dir = Path(args.output_dir)\n    output_dir.mkdir(parents=True, exist_ok=True)\n    \n    # Create data loaders\n    logger.info(\"Loading dataset...\")\n    train_loader, test_loader = create_data_loaders(\n        args.data_dir, args.batch_size, args.augment\n    )\n    \n    # Create model\n    if args.model == \"simple\":\n        model = SimpleMNISTNet()\n    else:\n        model = ImprovedMNISTNet()\n    \n    model.to(device)\n    \n    # Print model info\n    from mnist_classifier.utils import get_model_summary\n    model_info = get_model_summary(model)\n    logger.info(f\"Model info: {model_info}\")\n    \n    # Train model\n    history = train_model(model, train_loader, test_loader, args, device)\n    \n    # Save model\n    model_filename = f\"mnist_{args.model}_model.pth\"\n    model_path = output_dir / model_filename\n    \n    metadata = {\n        \"model_type\": args.model,\n        \"epochs\": args.epochs,\n        \"batch_size\": args.batch_size,\n        \"learning_rate\": args.learning_rate,\n        \"augmentation\": args.augment,\n        \"final_train_accuracy\": history['train_accuracies'][-1],\n        \"final_test_accuracy\": history['test_accuracies'][-1],\n        \"device\": device\n    }\n    \n    success = save_model(model, str(model_path), metadata)\n    if success:\n        logger.info(f\"Model saved to {model_path}\")\n    \n    # Plot training results\n    if args.plot_results:\n        plot_path = output_dir / f\"training_history_{args.model}.png\"\n        plot_training_history(history, str(plot_path))\n    \n    # Run evaluation\n    if args.evaluate:\n        logger.info(\"Running comprehensive evaluation...\")\n        eval_dir = output_dir / \"evaluation\"\n        generate_evaluation_report(model, test_loader, device, str(eval_dir))\n    \n    logger.info(\"Training completed successfully!\")\n    logger.info(f\"Final test accuracy: {history['test_accuracies'][-1]:.4f}\")\n\n\nif __name__ == \"__main__\":\n    main()"}