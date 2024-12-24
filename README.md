## **Project Overview**
This project demonstrates the development of a neural network for classifying handwritten digits from the MNIST dataset. The MNIST dataset is a popular benchmark in the field of machine learning and computer vision, consisting of grayscale images of digits (0–9).

Key tasks include:
- Loading and preprocessing the dataset.
- Building and training a neural network in PyTorch.
- Evaluating model performance on unseen test data.
- Enhancing performance through validation and architectural improvements.
- Saving the trained model for future use.

---

## **Features**
1. **Data Loading and Exploration**:
   - Utilizes `torchvision.datasets` to load the MNIST dataset.
   - Preprocesses data with normalization and tensor conversion.
   - Displays sample images to understand the dataset visually.
   - Splits the training data into training and validation subsets (80-20 split).

2. **Model Design and Training**:
   - Implements a feedforward neural network with multiple hidden layers.
   - Improves accuracy by upgrading to a Convolutional Neural Network (CNN).
   - Uses `CrossEntropyLoss` as the loss function and `Adam` optimizer for effective training.
   - Tracks training and validation loss/accuracy to monitor model performance.

3. **Model Evaluation**:
   - Achieves over 90% test accuracy.
   - Compares results with benchmarks on Yann LeCun’s MNIST research page.

4. **Model Saving**:
   - Saves the trained model parameters using `torch.save` for future reuse.

---

## **Repository Link**
This project is hosted on GitHub:  
[MNIST Handwritten Digits (Udacity)](https://github.com/dustinober1/MNIST_Handwritten_Digits_Udacity)

---

## **Getting Started**
Follow these steps to reproduce the project:

### **Prerequisites**
- Python 3.7+
- Required Libraries: 
  - PyTorch
  - torchvision
  - matplotlib
  - numpy
- Install dependencies via:
  ```bash
  pip install -r requirements.txt
  ```

### **Running the Notebook**
1. **Clone or Download the Repository**:
   ```bash
   git clone https://github.com/dustinober1/MNIST_Handwritten_Digits_Udacity.git
   cd MNIST_Handwritten_Digits_Udacity
   ```

2. **Run the Jupyter Notebook**:
   Open the `.ipynb` file in Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook mnist_project.ipynb
   ```

3. **Train the Model**:
   Execute all cells sequentially. Training progress and results will be displayed inline.

---

## **Details of Implementation**

### **1. Data Preprocessing**
- **Transforms**:
  - `ToTensor()`: Converts images to tensors.
  - `Normalize((0.5,), (0.5,))`: Normalizes pixel values to the range [-1, 1].
- **Dataset Splits**:
  - Training (80% of train dataset).
  - Validation (20% of train dataset).
  - Test dataset remains untouched for evaluation.

### **2. Model Architectures**
- **Feedforward Neural Network**:
  - A baseline model with three linear layers and ReLU activations.
  - Achieves over 90% test accuracy.

- **Convolutional Neural Network (CNN)**:
  - Uses convolutional and pooling layers for feature extraction.
  - Achieves improved accuracy and generalization.

### **3. Hyperparameter Tuning**
- Learning rate adjustments, deeper architectures, and more epochs were explored to optimize performance.

### **4. Evaluation Metrics**
- **Accuracy**: The percentage of correctly predicted digits.
- **Loss**: Tracks model performance during training and validation.

---

## **Results**
1. **Test Accuracy**: 
   - Baseline Feedforward Model: ~97%.
   - Convolutional Neural Network: Achieved improved performance with test accuracy >95%.

2. **Contextualization**:
   - Compared to benchmarks:
     - 88% (LeCun et al., 1998): Classical methods.
     - 99.65% (Ciresan et al., 2011): State-of-the-art CNNs.
   - This project demonstrates effective usage of PyTorch to build competitive models on MNIST.

---

## **Future Work**
1. Explore advanced architectures such as ResNet or MobileNet.
2. Experiment with data augmentation to enhance generalization.
3. Implement early stopping and learning rate scheduling.
4. Deploy the model as a web or mobile application for real-time digit recognition.

---

## **Acknowledgments**
- The MNIST dataset is provided by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges.
- Tutorials and documentation from PyTorch and torchvision were invaluable.
