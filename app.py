import os
os.system('pip install joblib')

import joblib
import streamlit as st

import numpy as np
from PIL import Image

# Load trained model, scaler, and PCA
model = joblib.load("mnist_model.pkl")  # Make sure the filename is correct
scaler = joblib.load("mnist_scaler.pkl")
pca = joblib.load("mnist_pca.pkl")

# Streamlit UI
st.title("MNIST Digit Classifier (ML + PCA)")
st.write("Upload a handwritten digit (28x28 grayscale) to predict its class.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28

    # Show image
    st.image(image, caption="Uploaded Image", width=150)

    # Convert image to numpy array
    image_array = np.array(image).reshape(1, -1)  # Flatten to 1D (784)
    
    # Normalize using the saved scaler
    image_scaled = scaler.transform(image_array)
    
    # Reduce dimensionality using PCA
    image_pca = pca.transform(image_scaled)

    # Predict digit
    prediction = model.predict(image_pca)[0]
    st.subheader(f"Predicted Digit: {prediction}")
