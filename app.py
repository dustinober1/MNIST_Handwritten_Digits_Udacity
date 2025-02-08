import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO

# =============================================================================
# Define the model architecture (ImprovedMNISTNet) as used during training
# =============================================================================
class ImprovedMNISTNet(nn.Module):
    def __init__(self):
        super(ImprovedMNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 256)  # First hidden layer with 256 neurons
        self.fc2 = nn.Linear(256, 128)      # Second hidden layer with 128 neurons
        self.fc3 = nn.Linear(128, 64)       # Third hidden layer with 64 neurons
        self.fc4 = nn.Linear(64, 10)        # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # Output logits (no activation here)
        return x

# =============================================================================
# Load the saved model
# =============================================================================
MODEL_PATH = "mnist_model.pth"

@st.cache_resource
def load_model():
    # Initialize the model using the improved architecture.
    model = ImprovedMNISTNet()
    # Load the state dictionary (weights) into the model.
    state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode.
    return model

# Load the model (this is cached so it won't reload on every interaction)
model = load_model()

# =============================================================================
# Image Preprocessing Function
# =============================================================================
def preprocess_image(image):
    """
    Preprocesses the image to match the training transforms:
      - Convert to grayscale (if not already)
      - Resize to 28x28 pixels
      - Convert to a tensor
      - Normalize pixel values to [-1, 1]
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure image is grayscale
        transforms.Resize((28, 28)),                    # Resize image to 28x28 pixels
        transforms.ToTensor(),                          # Convert image to tensor with values [0,1]
        transforms.Normalize((0.5,), (0.5,))            # Normalize to [-1,1]
    ])
    return transform(image).unsqueeze(0)  # Add a batch dimension

# =============================================================================
# Prediction Function
# =============================================================================
def predict_digit(image):
    """
    Predicts the digit from the input image.
    Returns the predicted digit and the confidence scores for each class.
    """
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    return prediction, probabilities.squeeze().numpy()

# =============================================================================
# Streamlit App Interface
# =============================================================================
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) or draw your own!")

# --------------------------
# Option 1: Upload an Image
# --------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    prediction, probabilities = predict_digit(image)
    
    st.write(f"**Predicted Digit:** {prediction}")
    st.write("**Confidence Scores:**")
    st.bar_chart(probabilities)

# --------------------------
# Option 2: Draw Your Own Digit
# --------------------------
st.write("Or draw your own digit below:")

# Try to import the drawing canvas module; if it's missing, display an error.
try:
    from streamlit_drawable_canvas import st_canvas

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="#000000",       # Background fill color for drawn areas
        stroke_width=10,
        stroke_color="#FFFFFF",     # Pen color (white)
        background_color="#000000", # Canvas background (black)
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        # Convert the canvas image (using only one channel) to a grayscale PIL image.
        drawn_image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8')).convert("L")
        resized_image = drawn_image.resize((28, 28))  # Resize to 28x28 pixels as expected by the model
        st.image(resized_image, caption="Drawn Image (Resized for Model)", use_column_width=False)
        
        if st.button("Predict Drawn Digit"):
            prediction, probabilities = predict_digit(resized_image)
            st.write(f"**Predicted Digit:** {prediction}")
            st.write("**Confidence Scores:**")
            st.bar_chart(probabilities)

except ModuleNotFoundError:
    st.error("The module 'streamlit-drawable-canvas' is not installed. Install it to enable the drawing feature.")