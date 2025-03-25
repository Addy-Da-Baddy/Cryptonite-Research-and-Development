import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

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

def train_new_model(dataset_path, output_folder):
    """Train new XGBoost and Neural Network models"""
    try:
        # Verify and load dataset
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        df = pd.read_csv(dataset_path)
        if 'class' not in df.columns:
            raise ValueError("Dataset must contain 'class' column")

        # Prepare data
        X = df.drop(columns=['class']).values
        y = df['class'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train XGBoost model
        print("\n[+] Training XGBoost model...")
        xgb_model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=10
        )
        xgb_model.fit(X_train, y_train)

        # Save XGBoost model
        os.makedirs(output_folder, exist_ok=True)
        xgb_path = os.path.join(output_folder, 'Malnet_XGB.pkl')
        joblib.dump(xgb_model, xgb_path)

        # Prepare hybrid features
        X_train_xgb = xgb_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        X_test_xgb = xgb_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        X_train_hybrid = np.hstack((X_train, X_train_xgb))
        X_test_hybrid = np.hstack((X_test, X_test_xgb))

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_hybrid, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_tensor = torch.tensor(X_test_hybrid, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

        # Initialize Neural Network
        model = MalwareNN(X_train_hybrid.shape[1]).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        # Training loop
        print("\n[+] Training Neural Network...")
        best_loss = float('inf')
        epochs = 150
        early_stop_patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor)
                scheduler.step(test_loss)

            # Early stopping check
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), os.path.join(output_folder, 'Malnet_NN.pth'))
            else:
                patience_counter += 1

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {loss.item():.4f} - "
                      f"Test Loss: {test_loss.item():.4f}")

            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model for evaluation
        model.load_state_dict(torch.load(os.path.join(output_folder, 'Malnet_NN.pth')))

        # Evaluate
        model.eval()
        with torch.no_grad():
            y_train_pred = (model(X_train_tensor) > 0.5).float().cpu().numpy()
            y_test_pred = (model(X_test_tensor) > 0.5).float().cpu().numpy()

        # Print results
        print("\n=== Training Results ===")
        print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))

        print(f"\n[+] Models successfully saved to {output_folder}")

    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")