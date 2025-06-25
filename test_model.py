import torch
import torchvision.models as models
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import argparse
import os

# Define ResNet18 for 1-channel input and 8 output classes
def get_resnet18_custom():
    model = models.resnet18(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model

# Step 1: Initialize the modified architecture
model = get_resnet18_custom().half()

# Step 2: Load the state dict
model_path = "final_model.pth"  # Use .pth extension
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict)

# Step 3: Set to eval mode
model.eval()

# Step 4: Report size
size_mb = os.path.getsize(model_path) / (1024 ** 2)
print(f"✅ Float16 ResNet18 model loaded (1-channel input, 8 classes). Size: {size_mb:.2f} MB")

# Step 5: Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Emotion class labels (update if needed)
class_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised', 'calm']

# Preprocess audio
def preprocess_audio(file_path, target_sample_rate=16000, n_mels=128):
    waveform, sample_rate = torchaudio.load(file_path)

    if sample_rate != target_sample_rate:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )(waveform)

    mel_db = T.AmplitudeToDB()(mel_spectrogram)
    mel_db = (mel_db - mel_db.mean()) / mel_db.std()

    return mel_db.unsqueeze(0).to(device).half()  # Shape: [1, 1, mel, time]

# Make prediction
def predict(file_path):
    input_tensor = preprocess_audio(file_path)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_index = outputs.argmax(dim=1).item()
        predicted_label = class_labels[predicted_index]
    print(f"🎯 Predicted Emotion: {predicted_label}")

# Main function with fallback input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Classification from Audio")
    parser.add_argument("audio_path", type=str, nargs="?", help="Path to input .wav file (e.g., python test_model.py audio.wav)")
    args = parser.parse_args()

    audio_path = args.audio_path

    while not audio_path or not os.path.isfile(audio_path):
        print(" Invalid or missing audio file path.")
        audio_path = input(" Please write in terminal in this way : python test_model.py path/to/audio.wav").strip()

    predict(audio_path)