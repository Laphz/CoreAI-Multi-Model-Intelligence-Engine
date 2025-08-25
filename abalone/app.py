import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import gdown
import os

# --------------------------
# Download Pickled Model & Scaler from Google Drive (if not exists)
# --------------------------
model_url = "https://drive.google.com/uc?id=1JdedP9I71lco0kNNkVWaQ-Rt0oONMas3"
scaler_url = "https://drive.google.com/uc?id=1TbRfJEzph9Ig6AQAe1spbfOy7GtHSgyB"

if not os.path.exists("model.pkl"):
    gdown.download(model_url, "model.pkl", quiet=False)

if not os.path.exists("scaler.pkl"):
    gdown.download(scaler_url, "scaler.pkl", quiet=False)

# --------------------------
# Load Pickled Model & Scaler
# --------------------------
with open("model.pkl", "rb") as f:
    rfr = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --------------------------
# Define Prediction Function
# --------------------------
def predict_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight):
    features = np.array([[Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight]])
    scaled_features = scaler.transform(features)
    prediction = rfr.predict(scaled_features)
    return prediction[0]

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="Abalone Age Prediction")
st.title('🐚 Abalone Age Prediction')

# ✅ Image code unchanged
img = Image.open('img.jpeg')
img = img.resize((650, 250))
st.image(img)

st.write("""
This app predicts the age of an abalone based on its physical measurements.
""")

# Input fields
col1, col2, col3 = st.columns(3)
col4, col5 = st.columns(2)

with col1:
    Sex = st.selectbox('Sex', ['Male', 'Female', 'Infant'])
    Sex = {'Male': 0, 'Female': 1, 'Infant': 2}[Sex]
    Height = st.number_input('Height', min_value=0.0, format="%.3f")

with col2:
    Length = st.number_input('Length', min_value=0.0, format="%.3f")
    Whole_weight = st.number_input('Whole Weight', min_value=0.0, format="%.3f")

with col3:
    Diameter = st.number_input('Diameter', min_value=0.0, format="%.3f")
    Shucked_weight = st.number_input('Shucked Weight', min_value=0.0, format="%.3f")

with col4:
    Viscera_weight = st.number_input('Viscera Weight', min_value=0.0, format="%.3f")

with col5:
    Shell_weight = st.number_input('Shell Weight', min_value=0.0, format="%.3f")

# Predict button
if st.button('🔮 Predict Age'):
    age = predict_age(Sex, Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight)
    st.success(f'The predicted age of the abalone is: **{age:.2f} years**')
