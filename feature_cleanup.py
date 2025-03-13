import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler

def load_and_clean_data(dataset_path='./data/ClaMP_Integrated-5184.csv'):

    warnings.filterwarnings("ignore")
    
    print(f"Loading dataset from: {dataset_path}")
    raw_dataset = pd.read_csv(dataset_path)
    
    categorical_cols = ['packer_type', 'fileinfo']
    for col in categorical_cols:
        print(f"{col}: {raw_dataset[col].unique()}")
    
    print("Performing one-hot encoding for categorical features...")
    raw_dataset = pd.get_dummies(raw_dataset, columns=['packer_type'], drop_first=True)
    
    print("Generating class distribution plot...")
    plot_class_distribution(raw_dataset)
    
    raw_dataset = raw_dataset.astype(int)
    
    print("Scaling numerical features...")
    num_cols = [col for col in raw_dataset.columns if raw_dataset[col].nunique() > 2]
    scaler = RobustScaler()
    raw_dataset[num_cols] = scaler.fit_transform(raw_dataset[num_cols])
    
    print(f"Dataset shape after cleaning: {raw_dataset.shape}")
    return raw_dataset

def plot_class_distribution(dataset):

    class_counts = dataset['class'].value_counts()
    plt.figure(figsize=(5, 3))
    class_counts.plot(kind='bar', color=['blue', 'red'])
    plt.title("Class Distribution (Benign vs Malware)")
    plt.xticks(ticks=[0, 1], labels=['Benign (0)', 'Malware (1)'])
    plt.ylabel("Count")
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def prepare_train_test_split(dataset, test_size=0.2, random_state=42):

    from sklearn.model_selection import train_test_split
    
    X = dataset.drop(columns=["class"])
    y = dataset["class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    dataset = load_and_clean_data()
    X_train, X_test, y_train, y_test = prepare_train_test_split(dataset)
    
    print("Saving processed datasets...")
    pd.concat([X_train, y_train], axis=1).to_csv('processed_train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('processed_test.csv', index=False)
    print("Processing complete!")