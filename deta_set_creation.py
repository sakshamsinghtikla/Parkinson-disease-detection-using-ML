import os
import numpy as np
import pandas as pd
import librosa

# Define paths
pd_folder = "PD_AH"  # Folder with Parkinson's samples
hc_folder = "HC_AH"  # Folder with Healthy samples

# Define required features
required_features = [
    'MDVP:PPQ', 'D2', 'RPDE', 'spread2', 'MDVP:RAP',
    'MDVP:APQ', 'PPE', 'Shimmer:APQ3', 'NHR', 'MDVP:Shimmer(dB)'
]

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
        return [features[feat] for feat in required_features]
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Collect data
data = []

# Process PD samples
for filename in os.listdir(pd_folder):
    if filename.endswith(".wav"):
        path = os.path.join(pd_folder, filename)
        features = extract_features(path)
        if features:
            data.append(features + [1])  # Label 1 for Parkinson's

# Process HC samples
for filename in os.listdir(hc_folder):
    if filename.endswith(".wav"):
        path = os.path.join(hc_folder, filename)
        features = extract_features(path)
        if features:
            data.append(features + [0])  # Label 0 for Healthy

# Create DataFrame
df = pd.DataFrame(data, columns=required_features + ['status'])

# Save to CSV
df.to_csv("newly_dataset.csv", index=False)
print("✅ Feature dataset saved as newly_dataset.csv")
