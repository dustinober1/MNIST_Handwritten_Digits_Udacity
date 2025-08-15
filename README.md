# 🔢 MNIST Handwritten Digit Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF6B6B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A modern, production-ready implementation of handwritten digit classification using deep learning. This project features a custom neural network trained on the MNIST dataset with a sleek Streamlit web interface for real-time predictions.

## ✨ Features

- **🧠 Custom Neural Network**: Improved architecture achieving 97%+ accuracy
- **🖼️ Interactive Web App**: Upload images or draw digits with real-time predictions
- **📊 Comprehensive Evaluation**: Detailed metrics, confusion matrices, and visualizations
- **🔧 Professional Codebase**: Modular design, type hints, comprehensive testing
- **📦 Easy Deployment**: Docker support and CLI tools
- **🚀 CI/CD Ready**: GitHub Actions integration

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [⚙️ Installation](#️-installation)
- [🎯 Usage](#-usage)
  - [Web Interface](#web-interface)
  - [Command Line](#command-line)
  - [Jupyter Notebook](#jupyter-notebook)
- [🧠 Model Architecture](#-model-architecture)
- [📊 Evaluation & Metrics](#-evaluation--metrics)
- [🧪 Testing](#-testing)
- [🐳 Docker Deployment](#-docker-deployment)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/mnist-digit-classifier.git
cd mnist-digit-classifier

# Install dependencies
pip install -r requirements.txt

# Train the model (optional - pre-trained model included)
python scripts/train.py --epochs 10 --evaluate

# Launch the web interface
streamlit run app.py
```

Open your browser to `http://localhost:8501` and start classifying digits!

## 📁 Project Structure

```
mnist-digit-classifier/
├── 📁 src/mnist_classifier/     # Core package
│   ├── __init__.py
│   ├── model.py                 # Neural network architectures  
│   ├── preprocessing.py         # Image preprocessing utilities
│   ├── utils.py                 # Helper functions
│   └── evaluation.py            # Model evaluation and metrics
├── 📁 scripts/                  # Command-line tools
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── 📁 tests/                    # Unit tests
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_utils.py
├── 📁 docs/                     # Documentation
├── app.py                       # Streamlit web interface
├── MNIST_Handwritten_Digits.ipynb # Jupyter notebook
├── requirements.txt             # Dependencies
├── Dockerfile                   # Container configuration
└── README.md                    # You are here!
```

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv mnist-env
   source mnist-env/bin/activate  # On Windows: mnist-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation
```bash
# Run tests to verify everything is working
pytest tests/ -v

# Check if CUDA is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🎯 Usage

### Web Interface

Launch the interactive Streamlit app:

```bash
streamlit run app.py
```

Features:
- 📤 **Upload images**: Drag and drop or browse for digit images
- ✏️ **Draw digits**: Use the interactive canvas to draw your own digits
- 📊 **Real-time predictions**: Get instant results with confidence scores
- 📈 **Visualization**: Beautiful charts showing prediction confidence

### Command Line

#### Training
```bash
# Basic training
python scripts/train.py

# Advanced training with custom parameters
python scripts/train.py --epochs 15 --batch-size 128 --learning-rate 0.0005 --augment --evaluate

# Train different model architectures
python scripts/train.py --model simple  # or --model improved
```

#### Evaluation
```bash
# Evaluate a trained model
python scripts/evaluate.py --model-path mnist_model.pth --output-dir results/
```

#### Available Options
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Training batch size (default: 64)
- `--learning-rate`: Learning rate for optimizer (default: 0.001)
- `--augment`: Enable data augmentation
- `--evaluate`: Run comprehensive evaluation after training
- `--device`: Use 'cpu', 'cuda', or 'auto'

### Jupyter Notebook

Explore the complete workflow in the interactive notebook:

```bash
jupyter notebook MNIST_Handwritten_Digits.ipynb
```

## 🧠 Model Architecture

### ImprovedMNISTNet

Our custom neural network achieves **97%+ accuracy** on the MNIST test set:

```
Input (28×28 pixels) → Flatten (784 features)
     ↓
FC Layer 1: 784 → 256 neurons + ReLU
     ↓  
FC Layer 2: 256 → 128 neurons + ReLU
     ↓
FC Layer 3: 128 → 64 neurons + ReLU
     ↓
Output Layer: 64 → 10 classes (digits 0-9)
```

**Key Features:**
- 🎯 **Xavier Weight Initialization**: For stable training
- 🔄 **ReLU Activation**: Fast and effective non-linearity
- 📉 **CrossEntropyLoss**: Optimized for multi-class classification
- ⚡ **Adam Optimizer**: Adaptive learning rate

### Preprocessing Pipeline

1. **Grayscale Conversion**: Ensures single-channel input
2. **Resize to 28×28**: Matches MNIST format
3. **Tensor Conversion**: PIL Image → PyTorch Tensor
4. **Normalization**: Pixel values scaled to [-1, 1] for stability
5. **Data Augmentation** (training only): Random rotations and translations

## 📊 Evaluation & Metrics

Comprehensive model evaluation includes:

- **📈 Accuracy Metrics**: Overall and per-class accuracy
- **🎯 Confusion Matrix**: Detailed misclassification analysis  
- **📊 Precision/Recall/F1**: Per-digit performance metrics
- **📉 Confidence Analysis**: Prediction certainty distributions
- **🔍 Error Analysis**: Common misclassification patterns

### Sample Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 97.2% |
| Precision | 97.1% |
| Recall | 97.2% |
| F1-Score | 97.1% |

Run evaluation to generate detailed reports:

```bash
python scripts/evaluate.py --model-path mnist_model.pth
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_model.py -v          # Model architecture tests
pytest tests/test_preprocessing.py -v   # Data preprocessing tests  
pytest tests/test_utils.py -v          # Utility function tests

# Generate coverage report
pytest tests/ --cov=src/mnist_classifier --cov-report=html
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t mnist-classifier .

# Run the container
docker run -p 8501:8501 mnist-classifier
```

### Docker Compose (with GPU support)

```bash
docker-compose up
```

Access the app at `http://localhost:8501`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/mnist-digit-classifier.git
cd mnist-digit-classifier

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### Code Quality

We use:
- **Black**: Code formatting
- **Flake8**: Linting  
- **Pytest**: Testing
- **Type hints**: For better code documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- **MNIST Dataset**: Created by Yann LeCun and collaborators
- **PyTorch**: The deep learning framework that powers our models
- **Streamlit**: For the amazing web app framework
- **scikit-learn**: For evaluation metrics and utilities

---

⭐ **Star this repository if you found it helpful!**

📧 **Questions?** Feel free to open an issue or reach out!

🚀 **Want to contribute?** Check out our [contributing guidelines](#-contributing)!