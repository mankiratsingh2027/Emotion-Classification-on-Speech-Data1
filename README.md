# 🎧 Audio Emotion Classification using XGBoost

A real-time speech emotion recognition system powered by XGBoost, wrapped in a beautiful and responsive Streamlit web application. Upload your `.wav` audio files and detect emotions instantly!

---

## 🚀 Key Features

✨ **Instant Emotion Prediction**  
🎙 Upload a `.wav` audio file  
🧠 Real-time prediction using MFCC, Chroma, and Mel features  
📈 Built with XGBoost for fast and accurate results  
🌐 Deployed as a user-friendly Streamlit web app  

---

## 🎯 Emotions Detected

The classifier is trained to recognize the following **8 emotions** from speech:

| Emotion   | Emoji  |
|-----------|--------|
| Angry     | 😠     |
| Calm      | 😌     |
| Disgust   | 🤢     |
| Fearful   | 😨     |
| Happy     | 😄     |
| Neutral   | 😐     |
| Sad       | 😢     |
| Surprise  | 😲     |

---

## 🧠 Model Architecture

- **Algorithm**: XGBoost (`XGBClassifier`)
- **Input Format**: `.wav` files (resampled to 16 kHz)
- **Features Used**:  
  - MFCC (Mel-Frequency Cepstral Coefficients)  
  - Chroma  
  - Mel Spectrogram  
- **Model File**: `xgb_model.json`

---

## 📁 Project Structure
README.md
app.py
Mankirat_Singh_emotion_classification.ipynb
requirements.txt
xgb_model.json

---

## Coefficient matrix
![image](https://github.com/user-attachments/assets/f168cd39-6732-4acc-a3d8-2ef6269bd671)

---

## Classification results:
XGBoost Classification Report:

| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.855     | 0.891  | 0.87     | 38.0    |
| 1     | 0.949     | 0.969  | 0.96     | 38.0    |
| 2     | 0.749     | 0.943  | 0.83     | 38.0    |
| 3     | 0.880     | 0.898  | 0.89     | 39.0    |
| 4     | 0.827     | 0.744  | 0.78     | 39.0    |
| 5     | 0.966     | 0.759  | 0.85     | 19.0    |
| 6     | 0.739     | 0.680  | 0.71     | 38.0    |
| 7     | 0.763     | 0.718  | 0.74     | 39.0    |
|       |           |        |          |         |
| *Accuracy*     |        |          | *0.83* |         |
| *Macro Avg*    | 0.841  | 0.825    | 0.83     | 288.0   |
| *Weighted Avg* | 0.833  | 0.829    | 0.83     | 288.0   |

---

## Streamlit app
https://emotion-classification-on-speech-data1-dp3ylgfnzayjgrngvkprrg.streamlit.app/
