import streamlit as st
import subprocess
import os
from PIL import Image
import signal

# Define your model directories
models = {
    "Keyword Extraction": "keyword_extraction",
    "Amazon Product": "amazone_product",
    "Job Placement": "job_placement_data",
    "Music Recommendation": "music_recommendation",
    "Resume Screening": "resume_screening",
    "Abalone": "abalone"
}

# Sidebar navigation
st.set_page_config(page_title="CoreAI-Multi-Model-Intelligence-Engine")
st.title("CoreAI-Multi-Model-Intelligence-Engine")

img = Image.open('main.jpeg')
img = img.resize((700, 400))
st.image(img)

selected_model = st.selectbox("Select the Model", list(models.keys()))

# Main directory
main_dir = os.path.dirname(os.path.abspath(__file__))

# Track the currently running process
if "process" not in st.session_state:
    st.session_state.process = None

def run_model_app(model_directory):
    app_path = os.path.join(main_dir, model_directory, "app.py")

    if not os.path.exists(app_path):
        st.error(f"App not found: {app_path}")
        return

    # Kill previous process if running
    if st.session_state.process is not None:
        try:
            os.kill(st.session_state.process.pid, signal.SIGTERM)
        except Exception:
            pass

    # Start new Streamlit process
    st.session_state.process = subprocess.Popen(
        ["streamlit", "run", app_path],
        cwd=os.path.join(main_dir, model_directory)
    )

# Button
col1, col2, col3, col4, col5 = st.columns(5)
with col3:
    if st.button("Go to model"):
        run_model_app(models[selected_model])
