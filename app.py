import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Malaria Cell Classifier", layout="centered")

# Header
st.title("Malaria Cell Classifier")
st.markdown("""
This application uses a Convolutional Neural Network (CNN) to analyze microscopic images of red blood cells. 
It classifies the cells as either **Parasitized (Infected)** or **Uninfected (Healthy)**.
""")

# Load the model (cached to improve performance)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("malaria_classifier.keras")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload a blood cell image (JPG or PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
    with col2:
        st.write("### Analysis Result")
        with st.spinner("Analyzing image..."):
            # Preprocess the image to match model input (128x128)
            img = image.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array)[0][0]
            
            # Output logic (Assuming > 0.5 is Uninfected)
            if prediction > 0.5:
                confidence = prediction * 100
                st.success("**Classification: Uninfected (Healthy)**")
                st.write(f"Confidence Score: {confidence:.2f}%")
            else:
                confidence = (1 - prediction) * 100
                st.error("**Classification: Parasitized (Infected)**")
                st.write(f"Confidence Score: {confidence:.2f}%")

# Footer disclaimer
st.markdown("---")
st.caption("Disclaimer: This model is for educational and research purposes only and is not intended for clinical diagnosis.")
