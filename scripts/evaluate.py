#!/usr/bin/env python3
"""Evaluation script for trained MNIST models."""

import argparse
import sys
from pathlib import Path
import torch
import torchvision
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mnist_classifier.model import ImprovedMNISTNet
from mnist_classifier.preprocessing import get_mnist_transforms
from mnist_classifier.utils import load_model, setup_logging
from mnist_classifier.evaluation import generate_evaluation_report


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained MNIST model")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model file")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Directory containing MNIST dataset")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"],
                       default="auto", help="Device to use for evaluation")
    
    return parser.parse_args()


def create_test_loader(data_dir: str, batch_size: int):
    """Create test data loader."""
    test_transform = get_mnist_transforms(train=False)
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return test_loader


def main():
    """Main evaluation function."""
    args = parse_arguments()
    logger = setup_logging()
    
    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device=device)
    
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return 1
    
    # Create test data loader
    logger.info("Loading test dataset...")
    test_loader = create_test_loader(args.data_dir, args.batch_size)
    
    # Run comprehensive evaluation
    logger.info("Running evaluation...")
    results = generate_evaluation_report(
        model, test_loader, device, save_dir=args.output_dir
    )
    
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())