# MNIST Digit Classifier with Streamlit

This project implements a neural network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). The classifier is trained using PyTorch, and a [Streamlit](https://streamlit.io/) web application is provided to allow users to either upload an image or draw a digit and receive real-time predictions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Streamlit App](#running-the-streamlit-app)
- [Usage](#usage)
- [Preprocessing & Model Architecture](#preprocessing--model-architecture)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The MNIST Digit Classifier project uses an improved fully connected neural network (`ImprovedMNISTNet`) to achieve high accuracy on the MNIST dataset. The project includes:

- **Model Training:** A complete training pipeline with data loading, preprocessing, model definition, training, and evaluation.
- **Web Interface:** A Streamlit app (`app.py`) that allows users to interact with the trained model by either uploading a digit image or drawing one on a canvas.
- **Model Saving & Loading:** After training, the model’s weights are saved to `mnist_model.pth` and later loaded by the Streamlit app for inference.

## Features

- **Neural Network Classifier:** Custom-built neural network with multiple fully connected layers.
- **Data Preprocessing:** Conversion of images to tensors, grayscale conversion, resizing to 28×28 pixels, and normalization.
- **Interactive Streamlit App:** Users can upload digit images or draw their own digits using an integrated drawing canvas.
- **Real-time Predictions:** The app displays the predicted digit along with a bar chart of confidence scores for each class.

## Project Structure

```plaintext
├── app.py                # Streamlit application for digit classification
├── train_model.ipynb     # Jupyter Notebook (or script) used to train the model
├── mnist_model.pth       # Saved model weights after training
├── requirements.txt      # List of Python dependencies
└── README.md             # This file
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/mnist-digit-classifier.git
   cd mnist-digit-classifier
   ```

2. **Set Up a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Make sure you have Python 3.7 or higher installed. Then install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Note:* The `requirements.txt` should include packages such as:
   - `torch`
   - `torchvision`
   - `streamlit`
   - `streamlit-drawable-canvas`
   - `Pillow`
   - `matplotlib`
   - etc.

## Training the Model

1. **Download the MNIST Dataset:**  
   The training script (or notebook) automatically downloads the MNIST dataset using `torchvision.datasets.MNIST` with `download=True`.

2. **Run the Training Notebook/Script:**

   If using the provided Jupyter Notebook (`train_model.ipynb`), run through the cells to:
   - Define transforms and create data loaders.
   - Define and train the `ImprovedMNISTNet` model.
   - Evaluate the model’s performance on the test set.
   - Save the trained model weights to `mnist_model.pth` using:

   ```python
   torch.save(model.state_dict(), "mnist_model.pth")
   ```

## Running the Streamlit App

Once the model is trained and the weights are saved, you can run the web interface.

1. **Start the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - **Upload Image:** Use the file uploader to select and display an image of a handwritten digit.
   - **Draw Digit:** Use the integrated drawing canvas to sketch a digit. Click the "Predict Drawn Digit" button to receive predictions.
   - The app will display the predicted digit along with a confidence score bar chart.

## Usage

- **Uploading an Image:**  
  Click on the "Upload a digit image" button, select a PNG/JPG/JPEG image, and wait for the prediction.

- **Drawing a Digit:**  
  Use your mouse or stylus to draw a digit on the canvas. Once satisfied with your drawing, press the "Predict Drawn Digit" button. The image is preprocessed (converted to grayscale, resized to 28×28 pixels, normalized) before being fed to the model.

## Preprocessing & Model Architecture

### Preprocessing
- **Grayscale Conversion:** Ensures the image has a single channel.
- **Resizing:** Images are resized to 28×28 pixels to match the MNIST dataset.
- **Normalization:** Pixel values are scaled to the range `[-1, 1]` using `transforms.Normalize((0.5,), (0.5,))`.

### Model Architecture

The `ImprovedMNISTNet` class is defined as follows:

- **Input Layer:** Flattens the 28×28 image to a 784-element vector.
- **Fully Connected Layers:**
  - **Layer 1:** 784 → 256 neurons (with ReLU activation).
  - **Layer 2:** 256 → 128 neurons (with ReLU activation).
  - **Layer 3:** 128 → 64 neurons (with ReLU activation).
  - **Output Layer:** 64 → 10 neurons (logits for each digit class 0–9).

The model uses the CrossEntropyLoss function and the Adam optimizer during training.

## Dependencies

- **PyTorch & Torchvision:** For building and training the neural network.
- **Streamlit:** To create the web interface.
- **Streamlit-Drawable-Canvas:** For the drawing interface.
- **Pillow:** For image processing.
- **Matplotlib:** For plotting and visualization.
- **NumPy:** For numerical operations.

Ensure all dependencies are installed via `pip install -r requirements.txt`.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- **MNIST Dataset:** Provided by Yann LeCun and his collaborators.
- **PyTorch:** For making deep learning research accessible.
- **Streamlit:** For enabling rapid development of interactive web applications.
