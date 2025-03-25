import argparse
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
import numpy as np
import joblib

class MalwareNN(nn.Module):
    def __init__(self, input_dim):
        super(MalwareNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def get_risk_tier(probability):
    if probability <= 20:
        return "🟢 SAFE - No threats detected."
    elif probability <= 40:
        return "🟡 LOW RISK - Unlikely to be malware, but caution advised."
    elif probability <= 60:
        return "🟠 MODERATE RISK - Suspicious file, might contain malware."
    elif probability <= 80:
        return "🔴 HIGH RISK - Likely malware, avoid execution."
    else:
        return "☠ CRITICAL THREAT - Very dangerous malware detected!"

def predict():
    try:
        df = pd.read_csv("preprocessed_extracted.csv")

        X = df.drop(columns=["class"]).values  

        xgb_model = joblib.load("Malnet_XGB.pkl")

        xgb_proba = xgb_model.predict_proba(X)[:, 1] 
        X_hybrid = np.hstack((X, xgb_proba.reshape(-1, 1)))  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nn_model = MalwareNN(X_hybrid.shape[1]).to(device)
        nn_model.load_state_dict(torch.load("Malnet_NN.pth", map_location=device))
        nn_model.eval()

        X_tensor = torch.tensor(X_hybrid, dtype=torch.float32).to(device)

        with torch.no_grad():
            nn_proba = nn_model(X_tensor).cpu().numpy().flatten()

        final_proba = nn_proba 

        safe_probability = final_proba[0] * 100
        malware_probability = 100 - safe_probability
        risk_level = get_risk_tier(malware_probability)

        print(f"\n[+] The file has a {malware_probability:.2f}% chance of being malware.")
        print(f"[+] Risk Assessment: {risk_level}")

    except Exception as e:
        print("[-] Error in prediction:", str(e))

if __name__ == "__main__":
    predict()