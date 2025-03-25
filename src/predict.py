import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MalwareNN(nn.Module):
    """Neural Network for malware detection"""
    def __init__(self, input_dim):
        super(MalwareNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

def get_risk_tier(probability):
    """Get human-readable risk assessment"""
    if probability <= 20:
        return "🟢 SAFE - No threats detected"
    elif probability <= 40:
        return "🟡 LOW RISK - Unlikely to be malware"
    elif probability <= 60:
        return "🟠 MODERATE RISK - Suspicious file"
    elif probability <= 80:
        return "🔴 HIGH RISK - Likely malware"
    else:
        return "☠ CRITICAL THREAT - Dangerous malware detected"

def load_models(model_folder):
    """Load XGBoost and Neural Network models"""
    try:
        # Verify model files exist
        required_files = ['Malnet_XGB.pkl', 'Malnet_NN.pth']
        for f in required_files:
            if not os.path.exists(os.path.join(model_folder, f)):
                raise FileNotFoundError(f"Missing model file: {f}")

        # Load XGBoost model
        xgb_model = joblib.load(os.path.join(model_folder, 'Malnet_XGB.pkl'))

        # Load Neural Network
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine input dimension by checking XGBoost model
        input_dim = xgb_model.n_features_in_ + 1  # +1 for XGB proba feature
        
        nn_model = MalwareNN(input_dim).to(device)
        nn_model.load_state_dict(torch.load(
            os.path.join(model_folder, 'Malnet_NN.pth'),
            map_location=device
        ))
        nn_model.eval()

        return xgb_model, nn_model, device

    except Exception as e:
        raise Exception(f"Model loading failed: {str(e)}")

def predict_with_default():
    """Predict using default models in models/default/"""
    predict_with_models("models/default")

def predict_with_custom(model_folder):
    """Predict using custom models in specified folder"""
    predict_with_models(model_folder)

def predict_with_models(model_folder):
    """Core prediction logic using specified models"""
    try:
        if not os.path.exists('preprocessed_extracted.csv'):
            raise FileNotFoundError("Preprocessed features not found. Run feature extraction first.")

        # Load preprocessed data
        df = pd.read_csv('preprocessed_extracted.csv')
        X = df.values  # Use all features

        # Load models
        xgb_model, nn_model, device = load_models(model_folder)

        # XGBoost prediction
        xgb_proba = xgb_model.predict_proba(X)[:, 1]
        X_hybrid = np.hstack((X, xgb_proba.reshape(-1, 1)))

        # Neural Network prediction
        X_tensor = torch.tensor(X_hybrid, dtype=torch.float32).to(device)
        with torch.no_grad():
            nn_proba = nn_model(X_tensor).cpu().numpy().flatten()

        # Combine predictions
        final_proba = nn_proba
        malware_prob = (1 - final_proba[0]) * 100  # Convert to malware probability

        # Display results
        print("\n" + "="*50)
        print(f"[+] Malware Probability: {malware_prob:.2f}%")
        print(f"[+] Risk Assessment: {get_risk_tier(malware_prob)}")
        print("="*50 + "\n")

    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")