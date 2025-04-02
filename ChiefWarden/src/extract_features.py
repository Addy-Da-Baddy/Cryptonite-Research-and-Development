import os
import pandas as pd
import pefile
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

FEATURES = [
    'e_cblp', 'e_cp', 'e_cparhdr', 'e_maxalloc', 'e_sp', 'e_lfanew',
    'NumberOfSections', 'CreationYear', 'FH_char0', 'FH_char1', 'FH_char2',
    'FH_char3', 'FH_char4', 'FH_char5', 'FH_char6', 'FH_char7', 'FH_char8',
    'FH_char9', 'FH_char10', 'FH_char11', 'FH_char12', 'FH_char13',
    'FH_char14', 'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
    'SizeOfInitializedData', 'SizeOfUninitializedData',
    'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase',
    'SectionAlignment', 'FileAlignment', 'MajorOperatingSystemVersion',
    'MinorOperatingSystemVersion', 'MajorImageVersion', 'MinorImageVersion',
    'MajorSubsystemVersion', 'MinorSubsystemVersion', 'SizeOfImage',
    'SizeOfHeaders', 'CheckSum', 'Subsystem', 'OH_DLLchar0', 'OH_DLLchar1',
    'OH_DLLchar2', 'OH_DLLchar3', 'OH_DLLchar4', 'OH_DLLchar5',
    'OH_DLLchar6', 'OH_DLLchar7', 'OH_DLLchar8', 'OH_DLLchar9',
    'OH_DLLchar10', 'SizeOfStackReserve', 'SizeOfStackCommit',
    'SizeOfHeapReserve', 'SizeOfHeapCommit', 'LoaderFlags', 'sus_sections',
    'non_sus_sections', 'packer', 'packer_type', 'E_text', 'E_data',
    'filesize', 'E_file', 'fileinfo', 'class'
]

def calculate_entropy(data):
    """Calculate entropy of data"""
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8))
    probabilities = counts / len(data)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def extract_features(filepath):
    """Extract features from PE file and save to CSV"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        pe = pefile.PE(filepath)
        extracted = {}

        # DOS Header features
        extracted.update({
            'e_cblp': pe.DOS_HEADER.e_cblp,
            'e_cp': pe.DOS_HEADER.e_cp,
            'e_cparhdr': pe.DOS_HEADER.e_cparhdr,
            'e_maxalloc': pe.DOS_HEADER.e_maxalloc,
            'e_sp': pe.DOS_HEADER.e_sp,
            'e_lfanew': pe.DOS_HEADER.e_lfanew,
        })

        # File Header features
        extracted.update({
            'NumberOfSections': pe.FILE_HEADER.NumberOfSections,
            'CreationYear': pe.FILE_HEADER.TimeDateStamp // (365 * 24 * 60 * 60) + 1970,
        })

        # File Header Characteristics (bit flags)
        for i in range(15):
            extracted[f'FH_char{i}'] = (pe.FILE_HEADER.Characteristics >> i) & 1

        # Optional Header features
        extracted.update({
            'MajorLinkerVersion': pe.OPTIONAL_HEADER.MajorLinkerVersion,
            'MinorLinkerVersion': pe.OPTIONAL_HEADER.MinorLinkerVersion,
            'SizeOfCode': pe.OPTIONAL_HEADER.SizeOfCode,
            'SizeOfInitializedData': pe.OPTIONAL_HEADER.SizeOfInitializedData,
            'SizeOfUninitializedData': pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            'AddressOfEntryPoint': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            'BaseOfCode': pe.OPTIONAL_HEADER.BaseOfCode,
            'BaseOfData': getattr(pe.OPTIONAL_HEADER, "BaseOfData", 0),
            'ImageBase': pe.OPTIONAL_HEADER.ImageBase,
            'SectionAlignment': pe.OPTIONAL_HEADER.SectionAlignment,
            'FileAlignment': pe.OPTIONAL_HEADER.FileAlignment,
            'MajorOperatingSystemVersion': pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            'MinorOperatingSystemVersion': pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            'MajorImageVersion': pe.OPTIONAL_HEADER.MajorImageVersion,
            'MinorImageVersion': pe.OPTIONAL_HEADER.MinorImageVersion,
            'MajorSubsystemVersion': pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            'MinorSubsystemVersion': pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            'SizeOfImage': pe.OPTIONAL_HEADER.SizeOfImage,
            'SizeOfHeaders': pe.OPTIONAL_HEADER.SizeOfHeaders,
            'CheckSum': pe.OPTIONAL_HEADER.CheckSum,
            'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
        })

        for i in range(11):
            extracted[f'OH_DLLchar{i}'] = (pe.OPTIONAL_HEADER.DllCharacteristics >> i) & 1

        extracted.update({
            'SizeOfStackReserve': pe.OPTIONAL_HEADER.SizeOfStackReserve,
            'SizeOfStackCommit': pe.OPTIONAL_HEADER.SizeOfStackCommit,
            'SizeOfHeapReserve': pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            'SizeOfHeapCommit': pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            'LoaderFlags': pe.OPTIONAL_HEADER.LoaderFlags,
        })

        # Section analysis
        sus_sections = sum(1 for s in pe.sections if b".text" not in s.Name)
        extracted.update({
            'sus_sections': sus_sections,
            'non_sus_sections': pe.FILE_HEADER.NumberOfSections - sus_sections,
        })

        extracted.update({
            'packer': 'Unknown',
            'packer_type': 'Unknown',
        })

        extracted.update({
            'E_text': calculate_entropy(pe.sections[0].get_data()) if len(pe.sections) > 0 else 0,
            'E_data': calculate_entropy(pe.sections[1].get_data()) if len(pe.sections) > 1 else 0,
            'filesize': os.path.getsize(filepath),
            'E_file': calculate_entropy(pe.__data__),
            'fileinfo': 1,  # Placeholder
            'class': 0,  # Default unknown
        })

        # Create DataFrame with correct feature order
        data = {feature: extracted.get(feature, 0) for feature in FEATURES}
        df = pd.DataFrame([data])

        # Save raw extracted features
        df.to_csv('extracted.csv', index=False)
        print("[+] Features extracted to extracted.csv")

        # Preprocess the features
        preprocess_data('extracted.csv')

    except Exception as e:
        raise Exception(f"Feature extraction failed: {str(e)}")

def preprocess_data(input_path, scaler_path="models/default/scaler.pkl"):
    """Preprocess and normalize the extracted features"""
    try:
        df = pd.read_csv(input_path)
        
        # Drop packer columns if present
        df.drop(columns=['packer_type', 'packer'], inplace=True, errors='ignore')

        # Fill missing values with median
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())

        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        # Load or create scaler
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[numeric_cols] = scaler.transform(df[numeric_cols])
        else:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

        # Save preprocessed data
        output_path = 'preprocessed_extracted.csv'
        df.to_csv(output_path, index=False)
        print(f"[+] Preprocessed data saved to {output_path}")

    except Exception as e:
        raise Exception(f"Preprocessing failed: {str(e)}")