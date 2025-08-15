"""Setup script for MNIST Classifier package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mnist-classifier",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modern PyTorch implementation of MNIST digit classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mnist-digit-classifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "pillow>=10.0.0",
        "streamlit>=1.28.0",
        "streamlit-drawable-canvas>=0.9.0",
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mnist-train=mnist_classifier.scripts.train:main",
            "mnist-evaluate=mnist_classifier.scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)