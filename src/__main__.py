import argparse
import os
import shutil
from .predict import predict_with_default, predict_with_custom
from .train_and_eval import train_new_model
from .extract_features import extract_features

def cleanup_temp_files():
    """Remove temporary files created during processing"""
    temp_files = ['extracted.csv', 'preprocessed_extracted.csv']
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"[-] Warning: Could not remove {file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="ChiefWarden Malware Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Predict malware using default models'
    )
    predict_parser.add_argument(
        'exe_path',
        help='Path to the executable file to analyze'
    )

    # PredictCustom command
    predict_custom_parser = subparsers.add_parser(
        'predictCustom',
        help='Predict using custom trained models'
    )
    predict_custom_parser.add_argument(
        'model_folder',
        help='Folder containing custom models (Malnet_NN.pth, Malnet_XGB.pkl, scaler.pkl)'
    )
    predict_custom_parser.add_argument(
        'exe_path',
        help='Path to the executable file to analyze'
    )

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train new models on custom dataset'
    )
    train_parser.add_argument(
        'dataset',
        help='Path to training dataset CSV file'
    )
    train_parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory to save trained models'
    )

    args = parser.parse_args()

    try:
        if args.command == 'predict':
            if not os.path.exists(args.exe_path):
                raise FileNotFoundError(f"Executable not found: {args.exe_path}")
            
            print("[+] Extracting features from executable...")
            extract_features(args.exe_path)
            
            print("[+] Running malware prediction...")
            predict_with_default()
            
        elif args.command == 'predictCustom':
            if not os.path.exists(args.exe_path):
                raise FileNotFoundError(f"Executable not found: {args.exe_path}")
            if not os.path.exists(args.model_folder):
                raise FileNotFoundError(f"Model folder not found: {args.model_folder}")
                
            required_models = ['Malnet_NN.pth', 'Malnet_XGB.pkl', 'scaler.pkl']
            for model in required_models:
                if not os.path.exists(os.path.join(args.model_folder, model)):
                    raise FileNotFoundError(f"Required model file missing: {model}")
            
            print("[+] Extracting features from executable...")
            extract_features(args.exe_path)
            
            print("[+] Running malware prediction with custom models...")
            predict_with_custom(args.model_folder)
            
        elif args.command == 'train':
            if not os.path.exists(args.dataset):
                raise FileNotFoundError(f"Dataset file not found: {args.dataset}")
                
            os.makedirs(args.output, exist_ok=True)
            
            print(f"[+] Training new models on {args.dataset}...")
            train_new_model(args.dataset, args.output)
            print(f"[+] Models successfully saved to {args.output}")
            
    except Exception as e:
        print(f"[-] Error: {str(e)}")
    finally:
        cleanup_temp_files()

if __name__ == "__main__":
    main()