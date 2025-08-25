import streamlit as st
import pickle
from PIL import Image
import gdown
import os

# --- Google Drive file IDs ---
df_file_id = "1YawG5VCRLot4bAbD9QPgIaf9dnqg9qSb"
similarity_file_id = "1rxJY0Vx-_Y6JzytfNFtqsw0gBpEUNZdl"

# Local cache names
df_path = "df.pkl"
similarity_path = "similarity.pkl"

# --- Download from Google Drive if not exists ---
if not os.path.exists(df_path):
    gdown.download(f"https://drive.google.com/uc?id={df_file_id}", df_path, quiet=False)

if not os.path.exists(similarity_path):
    gdown.download(f"https://drive.google.com/uc?id={similarity_file_id}", similarity_path, quiet=False)

# --- Load models ---
df = pickle.load(open(df_path, 'rb'))
similarity = pickle.load(open(similarity_path, 'rb'))

def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)

    return songs

# --- Streamlit App ---
st.set_page_config(page_title='Song Recommendation System')
st.title('Song Recommendation System')

# Keep your existing image logic
img = Image.open('img.jpeg')
img = img.resize((600, 400))
st.image(img)

# Dropdown for song selection
selected_song = st.selectbox("Select a song you like:", df['song'].values)

# Button to generate recommendations
if st.button("Recommend"):
    recommended_songs = recommendation(selected_song)
    st.subheader("Recommended Songs:")

    # Display recommendations in a grid with the headphone logo
    music_logo = 'Music_logo.png'  # local logo image (unchanged)

    cols_per_row = 3  # Number of columns per row
    for i in range(0, len(recommended_songs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, song in zip(cols, recommended_songs[i:i + cols_per_row]):
            with col:
                st.image(music_logo, width=100)  # ✅ unchanged image logic
                st.write(f"**{song}**")
