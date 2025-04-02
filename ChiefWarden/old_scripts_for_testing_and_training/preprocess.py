import pandas as pd
import argparse
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

def preprocess_data(input_path, scaler_path="scaler.pkl"):
    df = pd.read_csv(input_path)

    df.drop(columns=['packer_type', 'packer'], inplace=True, errors='ignore')

    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].median())

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        print(f"Loaded existing scaler from: {scaler_path}")
    else:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Created and saved new scaler to: {scaler_path}")

    output_path = "preprocessed.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the ClaMP dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to the ClaMP dataset CSV file.")
    args = parser.parse_args()

    preprocess_data(args.dataset_path)