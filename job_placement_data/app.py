import pandas as pd
import numpy as np
from PIL import Image
import pickle as pkl
import streamlit as st
import gdown
import os

# -----------------------------
# Custom CSS styling for input
# -----------------------------
st.markdown("""
    <style>
    .stTextInput input {
        font-size: 18px;
        height: 45px;
        width: 500px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Load model.pkl from Google Drive if not exists
# -----------------------------
model_file = "model.pkl"

if not os.path.exists(model_file):
    # 🔁 Replace with your own Google Drive FILE ID
    file_id = "1Nttd9SWfMXPT6mWoT2KXLv2QhWg81Sdc"
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_file, quiet=False)

# Load pickled model
lg = pkl.load(open(model_file, 'rb'))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Job Placement Predication Model")

# ✅ Do not change image logic
img = Image.open('img.jpeg')
st.image(img, width=600)

st.title('Job Placement Predication Model')

input_text = st.text_input("Enter all the features:")

if input_text:
    input_list = input_text.split(',')
    np_df = np.asarray(input_list, dtype=float)
    pred = lg.predict(np_df.reshape(1, -1))
    if pred[0] == 1:
        st.markdown("<p style='font-size:24px; font-weight:bold;'>Already Placed</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size:24px; font-weight:bold;'>Not Placed</h1>", unsafe_allow_html=True)
