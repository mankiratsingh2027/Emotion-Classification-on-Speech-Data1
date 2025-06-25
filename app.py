import streamlit as st
import librosa
import numpy as np
from xgboost import XGBClassifier
import tempfile
import warnings

# Silence warnings
warnings.filterwarnings("ignore")

# Load model
@st.cache_resource
def load_model(path="xgb_model.json"):
    model = XGBClassifier()
    model.load_model(path)
    return model

xgb_model = load_model()

# Emotion labels
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprise']

# Feature extraction: MFCC + Chroma + Mel
def extract_feature(file_path, n_mfcc=40, n_mels=128):
    y, sr = librosa.load(file_path, sr=None)

    # Resample to 16kHz
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels).T, axis=0)

    return np.hstack([mfcc, chroma, mel])

# Predict emotion
def predict_emotion(audio_path):
    features = extract_feature(audio_path)
    features = features.reshape(1, -1)
    pred_idx = xgb_model.predict(features)[0]
    return emotion_labels[pred_idx]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="🎧 XGBoost Emotion Classifier", layout="centered")

# Custom styles
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto Slab', serif;
        background-color: #f9f9f9;
    }

    h1 {
        color: #3f51b5;
        font-size: 40px;
        margin-bottom: 10px;
    }

    .stMarkdown p {
        font-size: 18px;
    }

    .stButton > button {
        background-color: #3f51b5;
        color: white;
        border: none;
        font-weight: bold;
        padding: 0.5em 1.2em;
        border-radius: 6px;
    }

    .stSuccess {
        font-size: 24px;
        font-weight: bold;
        color: green;
    }

    </style>
""", unsafe_allow_html=True)

st.title("🎙️ Emotion Detection from Audio")
st.markdown("Upload a `.wav` file and get the predicted **emotion** based on audio features.")

# Upload audio
uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Analyzing emotion..."):
        try:
            emotion = predict_emotion(tmp_path)
            st.success(f"🎯 Predicted Emotion: **{emotion.upper()}**")
        except Exception as e:
            st.error(f"❌ Error: {e}")
