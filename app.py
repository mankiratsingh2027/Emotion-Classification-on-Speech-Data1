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

# Custom styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Quicksand', sans-serif;
        background-color: #fdf6f0;
        color: #2e2e2e;
    }

    h1 {
        color: #ff6f61;
        font-size: 42px;
        margin-bottom: 0.2em;
    }

    .stMarkdown p {
        font-size: 18px;
        color: #444;
    }

    .stButton > button {
        background-color: #ff6f61;
        color: white;
        font-weight: bold;
        padding: 0.6em 1.4em;
        border-radius: 8px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #e85b50;
    }

    .stSuccess {
        font-size: 22px;
        font-weight: 600;
        color: #00897b;
    }

    .stSpinner > div {
        color: #ff6f61;
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
