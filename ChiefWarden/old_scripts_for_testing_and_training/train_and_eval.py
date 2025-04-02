
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

def train_and_evaluate(device):
    df = pd.read_csv("preprocessed.csv") 

    X = df.drop(columns=["class"]).values
    y = df["class"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining XGBoost: ")
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)

    joblib.dump(xgb_model, "Malnet_XGB.pkl")

    X_train_xgb = xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
    X_test_xgb = xgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)

    X_train_hybrid = np.hstack((X_train, X_train_xgb))
    X_test_hybrid = np.hstack((X_test, X_test_xgb))

    X_train_tensor = torch.tensor(X_train_hybrid, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test_hybrid, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    print("\nTraining NN: ")
    model = MalwareNN(X_train_hybrid.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
        if ((epoch+1)%10==0):
            tqdm.write(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

    torch.save(model.state_dict(), "Malnet_NN.pth")

    model.eval()
    with torch.no_grad():
        y_train_pred = (model(X_train_tensor) > 0.5).cpu().numpy()
        y_test_pred = (model(X_test_tensor) > 0.5).cpu().numpy()

    print("\nFinal Evaluation")
    print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate XGB + NN Hybrid model.")
    parser.add_argument("device", choices=["cpu", "cuda"], help="Device to use (cpu or cuda).")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    train_and_evaluate(device)