import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from xgboost import XGBClassifier
import joblib

def load_or_process_data(load_processed=False, dataset_path='./data/ClaMP_Integrated-5184.csv'):

    if load_processed:
        train_data = pd.read_csv('processed_train.csv')
        test_data = pd.read_csv('processed_test.csv')
        
        X_train = train_data.drop(columns=["class"])
        y_train = train_data["class"]
        X_test = test_data.drop(columns=["class"])
        y_test = test_data["class"]
    else:
        from feature_cleanup import load_and_clean_data, prepare_train_test_split
        dataset = load_and_clean_data(dataset_path)
        X_train, X_test, y_train, y_test = prepare_train_test_split(dataset)
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, params=None):

    if params is None:
        params = {
            'n_estimators': 100, 
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'use_label_encoder': False, 
            'eval_metric': 'logloss'
        }
    
    print("Training XGBoost model with parameters:")
    for param, value in params.items():
        print(f"  {param}: {value}")
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print("\n===== Model Evaluation =====")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred))
    
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
    plt.title('Training Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
    plt.title('Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sorted_idx = model.feature_importances_.argsort()
    top_features = sorted_idx[-20:]  # Show top 20 features
    plt.barh(range(len(top_features)), model.feature_importances_[top_features])
    plt.yticks(range(len(top_features)), X_train.columns[top_features])
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {"train_accuracy": train_accuracy, "test_accuracy": test_accuracy}

def save_model(model, filename='xgboost_model.pkl'):

    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    X_train, X_test, y_train, y_test = load_or_process_data(load_processed=True)
    
    model = train_xgboost_model(X_train, y_train)
    
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    save_model(model)
    
    print("\nBaseline XGBoost model training and evaluation complete!")