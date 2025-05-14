import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import joblib
import os

# Set page config
st.set_page_config(page_title="Parkinson's Detection from Voice", layout="centered")

# Load model and scaler
model_dict = joblib.load("parkinson_ensemble_model.pkl")
model = model_dict["Random Forest"]  # Choose from "XGBoost", "Random Forest", etc.
scaler = joblib.load("scaler.pkl")

# Features to extract
required_features = ['MDVP:PPQ', 'D2', 'RPDE', 'spread2', 'MDVP:RAP',
                     'MDVP:APQ', 'PPE', 'Shimmer:APQ3', 'NHR', 'MDVP:Shimmer(dB)']

# Feature extraction function
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, duration=3, offset=0.5)
        features = {
            'MDVP:PPQ': np.mean(librosa.feature.zero_crossing_rate(y)),
            'D2': np.var(y),
            'RPDE': np.mean(librosa.feature.spectral_flatness(y=y)),
            'spread2': np.mean(librosa.feature.spectral_bandwidth(y=y)),
            'MDVP:RAP': np.mean(librosa.feature.rms(y=y)),
            'MDVP:APQ': np.mean(librosa.feature.spectral_rolloff(y=y)),
            'PPE': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
            'Shimmer:APQ3': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            'NHR': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            'MDVP:Shimmer(dB)': np.std(y)
        }
        return np.array([features[feat] for feat in required_features]).reshape(1, -1)
    except Exception as e:
        print("Feature extraction failed:", e)
        return None

# Generate spectrogram
def display_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    st.subheader("Spectrogram")
    fig, ax = plt.subplots()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    st.pyplot(fig)

# Sidebar with logo
with st.sidebar:
    st.image("IMG_8445.jpg", width=70)  # <-- Your logo
    menu = st.selectbox("Menu", ["Predict", "About"])

# Main section
st.title("🎙️ Parkinson's Disease Detection from Voice")

if menu == "Predict":
    st.header("Upload a voice recording")
    audio_file = st.file_uploader("Choose an audio file (.wav, .m4a)", type=["wav", "m4a"])

    if audio_file:
        audio_path = f"temp_audio.{audio_file.name.split('.')[-1]}"
        with open(audio_path, "wb") as f:
            f.write(audio_file.read())

        # Show audio and spectrogram
        st.audio(audio_path)
        display_spectrogram(audio_path)

        # Feature extraction
        features = extract_features(audio_path)
        if features is not None:
            scaled_features = scaler.transform(features)
            st.write("Extracted Features:", features)

            prediction = model.predict(scaled_features)[0]

            if prediction == 1:
                st.success("✅ This voice sample does not indicate signs of Parkinson's Disease.")
            else:
                st.error("🧠 This voice sample indicates signs of Parkinson's Disease (PD).")

        os.remove(audio_path)

elif menu == "About":
    st.header("About the Project")
    st.markdown("""
**Parkinson’s Disease** is a neurodegenerative disorder that affects movement. One of the early signs includes subtle changes in voice—such as tremors, hoarseness, or lower volume—which can be picked up using audio analysis.

This model analyzes **speech patterns** and **biomedical voice features** from uploaded audio samples to detect whether a person may exhibit signs of Parkinson’s Disease (PD). It uses a machine learning classification model trained on voice samples from both PD patients and healthy individuals.

---

### 🧠 Methodology Overview:
""")
    
    st.image("PHOTO-2025-05-09-03-20-04.png", caption="📊 System Architecture: From audio input to prediction using features and classification.", use_column_width=True)

    st.markdown("""
---

### 🛠 How it Works:
1. **Input**: Uploads a short voice sample (e.g., vowel or sentence).
2. **Preprocessing**: Removes noise and prepares the signal.
3. **Feature Extraction**: Extracts phonation and articulation features using libraries like `librosa`.
4. **Feature Selection**: Selects most relevant biomedical indicators.
5. **Classification**: Predicts if the voice indicates Parkinson’s or not.

**Disclaimer**: This tool is for research and educational purposes only. It is not a replacement for medical diagnosis.
""")
