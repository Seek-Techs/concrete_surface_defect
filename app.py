import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import os
import json # Import json to potentially load history

# --- TensorFlow Configuration to Disable XLA ---
# This line disables XLA's JIT compilation, which often resolves tf2xla errors
tf.config.optimizer.set_jit(False) 
# Optional: If you suspect GPU issues, you can explicitly set visible devices to CPU
# tf.config.set_visible_devices([], 'GPU') 

# --- Streamlit Page Configuration (MUST BE FIRST Streamlit command) ---
st.set_page_config(
    page_title="Concrete Surface Defect Classifier",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Constants (Must match your training setup) ---
IMAGE_SIZE = 120
CHANNELS = 3
class_names = ["No Defect", "Defect"] # IMPORTANT: Ensure this order matches your dataset's inferred labels

# --- Model Loading ---
# Use st.cache_resource to load the model only once, improving performance
@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    try:
        # Load the model from the saved .h5 file
        model = tf.keras.models.load_model('concrete_surface_model.h5')
        return model
    except Exception as e:
        # These st. commands are now fine because set_page_config() has already been called
        st.error(f"Error loading model: {e}")
        st.info("Please ensure 'concrete_surface_model.h5' is in the same directory as this script.")
        st.stop() # Stop the app if model fails to load

model = load_model()

if model is None:
    st.stop() # Exit if model didn't load


# --- Prediction Function (Adapted from our previous discussion) ---
def predict(model, img):
    """
    Predicts the class and confidence for a single input image.
    This function expects a PIL Image object as input.

    Args:
        model: The trained Keras model.
        img: The input image as a PIL Image object.

    Returns:
        predicted_class (str): The human-readable predicted class name.
        confidence (int): The confidence percentage for the predicted class.
    """
    # Convert PIL Image to NumPy array
    img_array = np.array(img)
    
    # Ensure it's float32 for TensorFlow processing
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    # Resize and normalize to 0-1 range, just like in training
    # tf.image.resize handles resizing, and the division handles normalization.
    img_tensor = tf.image.resize(img_tensor, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = img_tensor / 255.0

    # Add the batch dimension (model expects input shape (batch_size, height, width, channels))
    img_array_processed = tf.expand_dims(img_tensor, 0)

    # Get predictions from the model
    predictions = model.predict(img_array_processed)
    
    # For binary classification with sigmoid output, predictions[0] is like [probability_of_class_1]
    prediction_prob = predictions[0][0] # Get the scalar probability for the positive class (Class 1)

    # Determine the predicted class based on a 0.5 threshold
    predicted_class_index = 1 if prediction_prob >= 0.5 else 0
    predicted_class = class_names[predicted_class_index]
    
    # Calculate confidence for the predicted class
    if predicted_class_index == 1:
        confidence = prediction_prob # If predicted Class 1, confidence is its probability
    else: # predicted_class_index == 0 (No Defect)
        confidence = 1 - prediction_prob # If predicted Class 0, confidence is (1 - probability_of_Class_1)
        
    confidence = round(100 * confidence) # Convert to percentage and round

    return predicted_class, confidence

# --- Streamlit UI Design (after set_page_config and model loading) ---

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stFileUploader {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        box-shadow: 4px 4px 10px rgba(0,0,0,0.3);
    }
    .stMarkdown h1 {
        color: #333333;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stMarkdown p {
        color: #555555;
        text-align: center;
    }
    .stAlert {
        border-radius: 8px;
    }
    .stImage > img {
        border-radius: 10px;
        box_shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏗️ Concrete Surface Defect Classifier")

st.markdown(
    """
    Upload an image of a concrete surface, and I'll predict if it contains a defect or not.
    """
)

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, JPEG, or PNG) to get a prediction."
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("") # Add some space

    if st.button("Predict Defect"):
        with st.spinner("Analyzing image..."):
            try:
                predicted_class, confidence = predict(model, image)
                st.success("Prediction Complete!")

                st.markdown(f"## **Prediction:** {predicted_class}")
                st.markdown(f"### **Confidence:** {confidence}%")

                # Updated condition for clarity and exact matching
                if predicted_class == class_names[1]: # Check if it's exactly the "Defect" class name
                    st.warning("Potential defect detected. Further inspection recommended.")
                else: # Implies predicted_class == class_names[0] ("No Defect")
                    st.info("No significant defect detected.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.info("Please try uploading a different image or ensure your model is correctly loaded.")

else:
    st.info("Awaiting image upload for defect detection.")

st.sidebar.header("About This App")
st.sidebar.markdown(
    """
    This application uses a Convolutional Neural Network (CNN) model trained with TensorFlow/Keras
    to classify concrete surfaces. It can help in quickly identifying potential defects.

    **Model Details:**
    - Input Image Size: 120x120 pixels
    - Channels: 3 (RGB)
    - Output: Binary classification (No Defect / Defect)
    """
)

st.sidebar.header("How to Use")
st.sidebar.markdown(
    """
    1.  **Upload an Image:** Click on "Choose an image..." and select a concrete surface image.
    2.  **Get Prediction:** Click the "Predict Defect" button.
    3.  **View Results:** The predicted class and confidence level will be displayed.
    """
)

# --- Model Performance Section ---
st.sidebar.header("Model Performance")

try:
    
    training_history = {
        'loss': [0.24421080946922302,
                     0.06975258141756058,
                     0.05530956760048866,
                     0.07344559580087662,
                     0.057269833981990814,
                     0.04271873086690903,
                     0.04000290855765343,
                     0.04415897652506828,
                     0.050962936133146286,
                     0.036991868168115616],
        'accuracy': [0.8852083086967468,
                     0.9777083396911621,
                     0.9829166531562805,
                     0.9745833277702332,
                     0.9802083373069763,
                     0.98458331823349,
                     0.9852083325386047,
                     0.9858333468437195,
                     0.981249988079071,
                     0.9860416650772095],
        'val_loss': [0.17940753698349,
                         0.060332801192998886,
                         0.06358304619789124,
                         0.06470923125743866,
                         0.06299105286598206,
                         0.04815433919429779,
                         0.05570437014102936,
                         0.05812826380133629,
                         0.09156458079814911,
                         0.1909899115562439], 
        'val_accuracy': [0.9375,
                         0.9809027910232544,
                         0.9826388955116272,
                         0.9704861044883728,
                         0.9756944179534912,
                         0.9861111044883728,
                         0.984375,
                         0.9791666865348816,
                         0.9618055820465088,
                         0.9461805820465088]
    }

    # Display key metrics
    st.sidebar.metric(
        label="Last Training Accuracy",
        value=f"{training_history['accuracy'][-1]*100:.2f}%"
    )
    st.sidebar.metric(
        label="Last Validation Accuracy",
        value=f"{training_history['val_accuracy'][-1]*100:.2f}%"
    )

    # Display accuracy plot
    st.sidebar.subheader("Accuracy over Epochs")
    st.sidebar.line_chart(
        {
            "Training Accuracy": training_history['accuracy'],
            "Validation Accuracy": training_history['val_accuracy']
        }
    )

    # Display loss plot
    st.sidebar.subheader("Loss over Epochs")
    st.sidebar.line_chart(
        {
            "Training Loss": training_history['loss'],
            "Validation Loss": training_history['val_loss']
        }
    )



except Exception as e:
    st.sidebar.error(f"Could not load/display performance metrics: {e}")
    st.sidebar.info("Ensure you have saved your model's training history (e.g., as JSON) and updated the loading path.")




