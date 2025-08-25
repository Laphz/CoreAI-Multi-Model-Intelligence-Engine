import pandas as pd
import nltk
import streamlit as st
import pickle
import gdown
import os
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Download tokenizer
nltk.download('punkt_tab')

# Tokenization (must match training)
sbs = SnowballStemmer('english')
def tokenization(txt):
    tokens = nltk.word_tokenize(txt.lower())
    txt = [sbs.stem(token) for token in tokens]
    return " ".join(txt)

# -----------------------------
# Load pickle from Google Drive
# -----------------------------
pickle_file = "product_search.pkl"

if not os.path.exists(pickle_file):
    # replace with your actual Google Drive file ID
    gdown.download("https://drive.google.com/uc?id=1pxCS6q4Na-V6j3wEfoxH37t4srilLV5k", pickle_file, quiet=False)

with open(pickle_file, "rb") as f:
    df, tfidf, tfidf_matrix = pickle.load(f)

# Search function
def search_product(search):
    stemmed_search = tokenization(search)
    search_vec = tfidf.transform([stemmed_search])
    similarity_scores = cosine_similarity(search_vec, tfidf_matrix).flatten()
    df['Similarity'] = similarity_scores
    return df.sort_values(by='Similarity', ascending=False).head(10)[['Title','Description','Category']]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Amazon Search Engine")
img = Image.open('img.jpeg')   # <-- unchanged as you said
st.image(img, width=600)
st.title('Search Engine and Product Recommendation System on Amazon Data')

search = st.text_input('Enter Product Name')
sumbitbtn = st.button('Search')

if sumbitbtn:
    res = search_product(search)
    st.write(res)
