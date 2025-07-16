import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
model = load_model('brain_tumor_classifier.h5')

# Define class names in the order used during training
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI image, and the model will predict the tumor type.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Show result
    st.success(f"🧠 Predicted Tumor Type: **{predicted_class.upper()}**")
