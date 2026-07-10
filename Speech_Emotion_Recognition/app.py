import os
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import streamlit as st
import tensorflow as tf


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main Title Styling */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        animation: fadeIn 0.8s ease-in;
    }

    /* Subtitle Styling */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #6c757d;
        margin-bottom: 30px;
        font-weight: 500;
        animation: fadeIn 1s ease-in 0.2s both;
    }

    /* Emotion Card - Enhanced */
    .emotion-card {
        border-radius: 25px;
        padding: 40px 30px;
        text-align: center;
        color: white;
        font-size: 3.5rem;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
        transform: translateY(0);
        transition: all 0.3s ease;
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }

    .emotion-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }

    .emotion-label {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 15px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* Confidence Bar Styling */
    .confidence-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }

    .confidence-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Probability Cards */
    .emotion-bar-container {
        background: white;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }

    .emotion-bar-container:hover {
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        transform: translateX(5px);
    }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }

    /* Audio Info Cards */
    .info-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.1);
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        color: #666;
        margin-top: 50px;
        padding-top: 30px;
        border-top: 2px solid #eee;
        font-size: 0.95rem;
    }

    .footer h4 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.3rem;
    }

    /* Block Container */
    .block-container {
        padding-top: 2rem;
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.05) 0%, rgba(255, 255, 255, 0) 100%);
    }

    /* Upload Area Styling */
    .upload-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }

    /* Success Message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }

    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes gradientShift {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    /* Responsive */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        .emotion-card {
            font-size: 2.5rem;
            padding: 30px 20px;
        }
    }

    /* Sidebar Styling */
    .sidebar-card {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }

    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 15px;
    }

    .sidebar-subtitle {
        font-size: 0.9rem;
        font-weight: 600;
        color: #555;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .dataset-file-item {
        background: white;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #764ba2;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.85rem;
        color: #333;
    }

    .dataset-file-item:hover {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        transform: translateX(5px);
        box-shadow: 0 3px 8px rgba(102, 126, 234, 0.15);
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Load Saved Objects
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/speech_emotion_dnn.keras"
    )


@st.cache_resource
def load_scaler():
    return joblib.load(
        "model/scaler.pkl"
    )


@st.cache_resource
def load_label_encoder():
    return joblib.load(
        "model/label_encoder.pkl"
    )


model = load_model()
scaler = load_scaler()
label_encoder = load_label_encoder()


# ---------------------------------------------------------
# Emotion Details
# ---------------------------------------------------------

emotion_details = {

    "Angry":(
        "😠",
        "#E74C3C",
        "The speaker exhibits strong emotional intensity commonly associated with anger."
    ),

    "Calm":(
        "😌",
        "#2ECC71",
        "The speech appears relaxed and emotionally stable."
    ),

    "Disgust":(
        "🤢",
        "#7D6608",
        "The speaker expresses vocal characteristics linked to disgust."
    ),

    "Fearful":(
        "😨",
        "#8E44AD",
        "The recording suggests signs of fear or anxiety."
    ),

    "Happy":(
        "😊",
        "#F1C40F",
        "The speech conveys positive and cheerful emotions."
    ),

    "Neutral":(
        "😐",
        "#95A5A6",
        "The speaker maintains a neutral emotional tone."
    ),

    "Sad":(
        "😢",
        "#3498DB",
        "The speech reflects sadness and low emotional energy."
    ),

    "Surprised":(
        "😲",
        "#E67E22",
        "The speaker demonstrates characteristics of surprise."
    )
}


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

def extract_features(audio_path):

    signal, sample_rate = librosa.load(
        audio_path,
        sr=None
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc, sample_rate, len(signal)


# ---------------------------------------------------------
# Get Dataset Files
# ---------------------------------------------------------

def get_dataset_files():
    """Get all .wav files from dataset directory"""
    dataset_path = Path("dataset")
    if dataset_path.exists():
        wav_files = []
        for actor_dir in sorted(dataset_path.glob("Actor_*")):
            if actor_dir.is_dir():
                actor_name = actor_dir.name  # e.g., "Actor_01"
                for wav_file in sorted(actor_dir.glob("*.wav")):
                    wav_files.append((wav_file.name, str(wav_file), actor_name))
        return wav_files
    return []


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("🎤 SER Dashboard")
st.sidebar.divider()

# Model Information Section
with st.sidebar.expander("🧠 Model Architecture", expanded=True):
    st.write("**Type:** Deep Neural Network (DNN)")
    st.write("**Framework:** TensorFlow/Keras")
    st.write("**Input Features:** 40 MFCC Coefficients")

# Dataset Information Section
with st.sidebar.expander("📊 Dataset Info", expanded=True):
    st.write("**Source:** RAVDESS Dataset")
    st.write("**Actors:** 24 Professional Actors")
    st.write("**Emotions:** 8 Emotion Classes")

# Emotion Classes Section
with st.sidebar.expander("😊 Emotion Classes", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.write("😠 Angry")
        st.write("😌 Calm")
        st.write("🤢 Disgust")
        st.write("😨 Fearful")
    with col2:
        st.write("😊 Happy")
        st.write("😐 Neutral")
        st.write("😢 Sad")
        st.write("😲 Surprised")

st.sidebar.divider()

# About Section
with st.sidebar.expander("ℹ️ About", expanded=True):
    st.write("This AI model detects emotions from speech recordings using advanced audio feature extraction and deep learning techniques.")


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------

st.markdown("""
<div class="main-title">
🎤 Speech Emotion Recognition
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Upload a <b>.wav</b> speech recording and let the AI detect the speaker's emotion with precision.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# File Upload & Dataset Selection
# ---------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

# Create tabs for upload and dataset selection
tab1, tab2 = st.tabs(["📤 Upload File", "📁 From Dataset"])

uploaded_file = None
selected_dataset_file = None

with tab1:
    st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 15px; margin-bottom: 20px;">
    <div style="font-size: 1.1rem; color: #555; font-weight: 500;">Choose a .wav file from your computer</div>
</div>
""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "📂 Upload Speech Audio (.wav)",
        type=["wav"],
        key="upload_file"
    )

with tab2:
    dataset_files = get_dataset_files()
    if dataset_files:
        st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 15px; margin-bottom: 20px;">
    <div style="font-size: 1.1rem; color: #555; font-weight: 500;">Select an audio file from the RAVDESS dataset</div>
</div>
""", unsafe_allow_html=True)
        
        # Group files by actor
        actors_dict = {}
        for file_name, file_path, actor_name in dataset_files:
            if actor_name not in actors_dict:
                actors_dict[actor_name] = []
            actors_dict[actor_name].append((file_name, file_path))
        
        # Select actor
        actor_list = sorted(actors_dict.keys())
        selected_actor = st.selectbox(
            "Select an Actor:",
            actor_list,
            key="actor_select"
        )
        
        # Select file from that actor
        files_for_actor = actors_dict[selected_actor]
        file_names = [f[0] for f in files_for_actor]
        
        selected_file_name = st.selectbox(
            "Select an Audio File:",
            file_names,
            key="file_select"
        )
        
        # Get the full path
        selected_idx = file_names.index(selected_file_name)
        selected_dataset_file = files_for_actor[selected_idx][1]
        
        st.markdown(f"""
<div style="padding: 15px; background: #f0f2f6; border-radius: 10px; border-left: 4px solid #667eea; margin-top: 15px;">
    <div style="font-size: 0.85rem; color: #666;"><b>Selected File:</b> {selected_file_name}</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("📁 No dataset files found. Please check if the dataset directory exists.")


# Determine which file to use
audio_file_to_process = uploaded_file if uploaded_file is not None else (selected_dataset_file if selected_dataset_file else None)

# ---------------------------------------------------------
# Audio Preview and Prediction
# ---------------------------------------------------------

if audio_file_to_process is not None:

    st.markdown("""
<div class="section-header">🎧 Audio Preview</div>
""", unsafe_allow_html=True)

    # Handle both uploaded files and dataset files
    if uploaded_file is not None:
        st.audio(uploaded_file)
    else:
        st.audio(selected_dataset_file)

    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Emotion", use_container_width=True, type="primary"):

            with st.spinner("Analyzing speech..."):

                # Handle file path for both uploaded and dataset files
                if uploaded_file is not None:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".wav"
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        temp_path = tmp.name
                    cleanup_needed = True
                else:
                    temp_path = selected_dataset_file
                    cleanup_needed = False

                try:

                    # Feature Extraction
                    # --

                    features, sample_rate, signal_length = extract_features(
                        temp_path
                    )

                    duration = signal_length / sample_rate

                    features = features.reshape(1, -1)

                    features = scaler.transform(features)

                    # Prediction
                    # --

                    prediction = model.predict(
                        features,
                        verbose=0
                    )

                    predicted_index = np.argmax(prediction)

                    predicted_emotion = label_encoder.inverse_transform(
                        [predicted_index]
                    )[0]

                    confidence = prediction[0][predicted_index] * 100

                    emoji, color, description = emotion_details[
                        predicted_emotion
                    ]

                    # Success Message
                    # --

                    st.markdown("""
<div class="success-message">
✅ Emotion Recognized Successfully!
</div>
""", unsafe_allow_html=True)

                    # Emotion Card
                    # --

                    st.markdown(
                        f"""
<div class="emotion-card"
style="background: linear-gradient(135deg, {color}dd 0%, {color} 100%);">
{emoji}
<div class="emotion-label">{predicted_emotion}</div>
</div>
""",
                        unsafe_allow_html=True
                    )

                    st.markdown(f"<p style='text-align: center; font-size: 1.1rem; color: #555; margin: 15px 0;'>{description}</p>", unsafe_allow_html=True)

                    st.divider()

                    # Confidence
                    # --

                    st.markdown("""
<div class="section-header">🎯 Prediction Confidence</div>
""", unsafe_allow_html=True)

                    st.markdown(f"""
<div class="confidence-section">
    <div style="text-align: center;">
        <div class="confidence-value">{confidence:.1f}%</div>
        <p style="color: #666; margin-top: 10px; font-weight: 500;">High confidence prediction</p>
    </div>
</div>
""", unsafe_allow_html=True)

                    st.progress(float(confidence / 100))

                    st.divider()

                    # Probability Scores
                    # --

                    st.markdown("""
<div class="section-header">📊 Emotion Probabilities</div>
""", unsafe_allow_html=True)

                    probabilities = prediction[0]

                    for emotion, probability in zip(
                        label_encoder.classes_,
                        probabilities
                    ):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"""
<div class="emotion-bar-container">
    <b>{emotion}</b>
    <div style="margin-top: 8px;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); height: 8px; border-radius: 10px; width: {probability*100}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)
                        with col2:
                            st.metric("", f"{probability * 100:.1f}%", label_visibility="collapsed")

                    st.divider()

                    # Audio Information
                    # --

                    st.markdown("""
<div class="section-header">🎵 Audio Information</div>
""", unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3, gap="medium")

                    with col1:
                        st.markdown(f"""
<div class="info-card">
    <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">⏱️ Duration</div>
    <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{duration:.2f}<span style="font-size: 1rem;"> sec</span></div>
</div>
""", unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
<div class="info-card">
    <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">📊 Sample Rate</div>
    <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">{sample_rate}<span style="font-size: 1rem;"> Hz</span></div>
</div>
""", unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
<div class="info-card">
    <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">🔊 MFCC Features</div>
    <div style="font-size: 1.8rem; font-weight: 700; color: #667eea;">40</div>
</div>
""", unsafe_allow_html=True)

                finally:

                    if cleanup_needed and os.path.exists(temp_path):
                        os.remove(temp_path)

else:

    st.markdown("""
<div style="text-align: center; padding: 60px 40px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(255, 255, 255, 0) 100%); border-radius: 15px; border: 2px dashed #667eea; margin-top: 30px;">
    <div style="font-size: 3rem; margin-bottom: 20px;">🎵</div>
    <div style="font-size: 1.5rem; font-weight: 700; color: #333; margin-bottom: 10px;">Ready to Analyze</div>
    <div style="color: #666; font-size: 1rem; line-height: 1.6;">
        Choose one of two options:
        <br><br>
        <b style="color: #667eea;">📤 Upload File:</b> Select and upload an audio file from your computer<br>
        <b style="color: #667eea;">📁 From Dataset:</b> Choose an audio file from the RAVDESS dataset
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.markdown(
    """
<div class="footer">

<h4 style="font-size: 1.5rem; margin-bottom: 15px;">🎤 Speech Emotion Recognition</h4>

<p style="margin: 15px 0; color: #666; font-size: 0.95rem;">
Built with ❤️ using<br>
<span style="display: inline-block; margin: 8px 5px; padding: 5px 12px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; font-weight: 600; color: #667eea;">TensorFlow</span>
<span style="display: inline-block; margin: 8px 5px; padding: 5px 12px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; font-weight: 600; color: #667eea;">Librosa</span>
<span style="display: inline-block; margin: 8px 5px; padding: 5px 12px; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)); border-radius: 20px; font-weight: 600; color: #667eea;">Streamlit</span>
</p>

<hr style="margin: 20px 0; border: none; border-top: 2px solid #eee;">

<p style="color: #667eea; font-weight: 600; margin-top: 15px;">
CodeAlpha Machine Learning Internship
</p>

</div>
""",
    unsafe_allow_html=True
)
