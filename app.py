import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

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
def load_model() -> Optional[ImprovedMNISTNet]:
    """Load the trained MNIST model with error handling."""
    try:
        if not Path(MODEL_PATH).exists():
            st.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
            return None
        
        model = ImprovedMNISTNet()
        state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logging.error(f"Model loading failed: {e}")
        return None

# Load the model with error handling
model = load_model()
if model is None:
    st.stop()

# =============================================================================
# Image Preprocessing Function
# =============================================================================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocesses the image to match the training transforms.
    
    Args:
        image: PIL Image to preprocess
        
    Returns:
        Preprocessed tensor ready for model input
        
    Raises:
        ValueError: If image preprocessing fails
    """
    try:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# =============================================================================
# Prediction Function
# =============================================================================
def predict_digit(image: Image.Image, model: ImprovedMNISTNet) -> Tuple[int, np.ndarray]:
    """
    Predicts the digit from the input image.
    
    Args:
        image: PIL Image containing handwritten digit
        model: Trained MNIST model
        
    Returns:
        Tuple of (predicted_digit, confidence_scores)
        
    Raises:
        ValueError: If prediction fails
    """
    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
        return prediction, probabilities.squeeze().numpy()
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

# =============================================================================
# Streamlit App Interface
# =============================================================================
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) or draw your own to see AI in action!")

# Add sidebar with information
with st.sidebar:
    st.header("About")
    st.write("""
    This app uses a neural network trained on the MNIST dataset 
    to classify handwritten digits (0-9).
    
    **Model Architecture:**
    - 4-layer fully connected network
    - 256 → 128 → 64 → 10 neurons
    - ReLU activation functions
    - Achieves ~97%+ accuracy
    """)
    
    st.header("Tips for better results")
    st.write("""
    - Draw digits clearly and large
    - Use white on black background
    - Center the digit in the canvas
    - Upload high-contrast images
    """)

# --------------------------
# Option 1: Upload an Image
# --------------------------
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            prediction, probabilities = predict_digit(image, model)
        
        st.success(f"**Predicted Digit:** {prediction}")
        st.write("**Confidence Scores:**")
        
        # Create a more informative chart
        import pandas as pd
        confidence_df = pd.DataFrame({
            'Digit': range(10),
            'Confidence': probabilities
        })
        st.bar_chart(confidence_df.set_index('Digit'))
        
        # Show confidence percentage for predicted digit
        confidence_pct = probabilities[prediction] * 100
        st.info(f"Confidence: {confidence_pct:.1f}%")
        
    except Exception as e:
        st.error(f"Error processing uploaded image: {str(e)}")
        logging.error(f"Upload processing failed: {e}")

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
        st.image(resized_image, caption="Drawn Image (Resized for Model)", use_container_width=False)
        
        if st.button("Predict Drawn Digit"):
            try:
                with st.spinner("Analyzing drawing..."):
                    prediction, probabilities = predict_digit(resized_image, model)
                
                st.success(f"**Predicted Digit:** {prediction}")
                st.write("**Confidence Scores:**")
                
                # Create a more informative chart
                import pandas as pd
                confidence_df = pd.DataFrame({
                    'Digit': range(10),
                    'Confidence': probabilities
                })
                st.bar_chart(confidence_df.set_index('Digit'))
                
                # Show confidence percentage
                confidence_pct = probabilities[prediction] * 100
                st.info(f"Confidence: {confidence_pct:.1f}%")
                
            except Exception as e:
                st.error(f"Error processing drawn image: {str(e)}")
                logging.error(f"Drawing processing failed: {e}")

except ModuleNotFoundError:
    st.error("""
    The module 'streamlit-drawable-canvas' is not installed. 
    
    To enable the drawing feature, install it with:
    ```
    pip install streamlit-drawable-canvas
    ```
    """)
except Exception as e:
    st.error(f"Error initializing drawing canvas: {str(e)}")
    logging.error(f"Canvas initialization failed: {e}")