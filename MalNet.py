import torch 
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

from feature_cleanup import load_and_clean_data, prepare_train_test_split

class MalwareDNN(nn.Module):
    def __init__(self, input_dim):
        super(MalwareDNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Linear(32, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        classification = self.classifier(features).squeeze(1)
        return features, classification

def load_or_process_data(load_processed=False, dataset_path='./data/ClaMP_Integrated-5184.csv'):
    if load_processed:
        train_data = pd.read_csv('processed_train.csv')
        test_data = pd.read_csv('processed_test.csv')
        
        X_train = train_data.drop(columns=["class"])
        Y_train = train_data["class"]
        X_test = test_data.drop(columns=["class"])
        Y_test = test_data["class"]
    else:
        dataset = load_and_clean_data(dataset_path)
        X_train, X_test, Y_train, Y_test = prepare_train_test_split(dataset)

    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    import pandas as pd

    X_train, X_test, Y_train, Y_test = load_or_process_data(load_processed=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    Y_test_tensor = torch.tensor(Y_test.values, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    dnn_model = MalwareDNN(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(dnn_model.parameters(), lr=0.001)

    epochs = 20
    dnn_model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        epoch_loss = 0.0
        batches = 0
        
        for batch_X, batch_Y in loop:
            batch_Y = batch_Y.float()
            optimizer.zero_grad()
            features, outputs = dnn_model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            epoch_loss += loss.item()
            batches += 1
            
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss/batches:.4f}")

    print("DNN training complete, extracting features...")
    
    with torch.no_grad():
        dnn_model.eval()
        train_features, train_outputs = dnn_model(X_train_tensor)
        test_features, test_outputs = dnn_model(X_test_tensor)

        train_features = train_features.cpu().numpy()
        test_features = test_features.cpu().numpy()
        
        train_outputs = torch.sigmoid(train_outputs).cpu().numpy()
        test_outputs = torch.sigmoid(test_outputs).cpu().numpy()
        
        train_preds = (train_outputs > 0.5).astype(int)
        test_preds = (test_outputs > 0.5).astype(int)
    
    print("DNN Evaluation:")
    print(f"DNN Train Accuracy: {accuracy_score(Y_train, train_preds):.4f}")
    print(f"DNN Test Accuracy: {accuracy_score(Y_test, test_preds):.4f}")
    print("DNN Train Report:")
    print(classification_report(Y_train, train_preds))
    print("DNN Test Report:")
    print(classification_report(Y_test, test_preds))
    
    torch.save(dnn_model.state_dict(), "dnn_model.pth")
    
    print("Creating ensemble with XGBoost...")
    X_train_combined = np.hstack((train_features, X_train.values))
    X_test_combined = np.hstack((test_features, X_test.values))

    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb_model.fit(X_train_combined, Y_train)
    
    Y_train_pred = xgb_model.predict(X_train_combined)
    train_accuracy = accuracy_score(Y_train, Y_train_pred)

    Y_test_pred = xgb_model.predict(X_test_combined)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)

    print(f"Ensemble Train Accuracy: {train_accuracy:.4f}")
    print(f"Ensemble Test Accuracy: {test_accuracy:.4f}")

    print("Ensemble Train Set Performance:")
    print(classification_report(Y_train, Y_train_pred))

    print("Ensemble Test Set Performance:")
    print(classification_report(Y_test, Y_test_pred))
    
    joblib.dump(xgb_model, "ensemble_xgb_model.pkl")
    
    print("Models saved. Training complete!")