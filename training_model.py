import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
file_path = "newly_dataset.csv"  # Update with your file path
df = pd.read_csv(file_path)

# Drop the 'name' column if present
df = df.drop(columns=['name'], errors='ignore')

# Separate features and target variable
X = df.drop(columns=['status'])
y = df['status']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Select top 10 most important features
important_features = ['MDVP:PPQ', 'D2', 'RPDE', 'spread2', 'MDVP:RAP', 'MDVP:APQ', 'PPE', 'Shimmer:APQ3', 'NHR', 'MDVP:Shimmer(dB)']
X_train_reduced = X_train[important_features]
X_test_reduced = X_test[important_features]

# Standardize the dataset
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

# Train models
models = {
      "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, n_jobs=-1, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
      
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results[name] = {
        "Accuracy": accuracy,
        "Confusion Matrix": cm,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-score": report['1']['f1-score']
    }
    
     # Display confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    gc.collect() 

    # Display results
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {report['1']['precision']:.4f}")
    print(f"Recall: {report['1']['recall']:.4f}")
    print(f"F1-score: {report['1']['f1-score']:.4f}")
# ✅ Save the trained models AFTER fitting
joblib.dump(models, "parkinson_ensemble_model.pkl")
print("✅ Trained models saved.")

# After fitting the scaler
joblib.dump(scaler, "scaler.pkl")

    