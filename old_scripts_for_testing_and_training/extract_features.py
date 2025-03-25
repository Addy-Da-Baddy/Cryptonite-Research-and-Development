import argparse
import pandas as pd
import pefile
import numpy as np
import subprocess

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

def extract_features(filepath):
    try:
        pe = pefile.PE(filepath)
        
        extracted = {
            "e_cblp": pe.DOS_HEADER.e_cblp,
            "e_cp": pe.DOS_HEADER.e_cp,
            "e_cparhdr": pe.DOS_HEADER.e_cparhdr,
            "e_maxalloc": pe.DOS_HEADER.e_maxalloc,
            "e_sp": pe.DOS_HEADER.e_sp,
            "e_lfanew": pe.DOS_HEADER.e_lfanew,
            "NumberOfSections": pe.FILE_HEADER.NumberOfSections,
            "CreationYear": pe.FILE_HEADER.TimeDateStamp // (365 * 24 * 60 * 60) + 1970, 
        }

        for i in range(15):
            extracted[f"FH_char{i}"] = (pe.FILE_HEADER.Characteristics >> i) & 1

        extracted.update({
            "MajorLinkerVersion": pe.OPTIONAL_HEADER.MajorLinkerVersion,
            "MinorLinkerVersion": pe.OPTIONAL_HEADER.MinorLinkerVersion,
            "SizeOfCode": pe.OPTIONAL_HEADER.SizeOfCode,
            "SizeOfInitializedData": pe.OPTIONAL_HEADER.SizeOfInitializedData,
            "SizeOfUninitializedData": pe.OPTIONAL_HEADER.SizeOfUninitializedData,
            "AddressOfEntryPoint": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
            "BaseOfCode": pe.OPTIONAL_HEADER.BaseOfCode,
            "BaseOfData": getattr(pe.OPTIONAL_HEADER, "BaseOfData", 0), 
            "ImageBase": pe.OPTIONAL_HEADER.ImageBase,
            "SectionAlignment": pe.OPTIONAL_HEADER.SectionAlignment,
            "FileAlignment": pe.OPTIONAL_HEADER.FileAlignment,
            "MajorOperatingSystemVersion": pe.OPTIONAL_HEADER.MajorOperatingSystemVersion,
            "MinorOperatingSystemVersion": pe.OPTIONAL_HEADER.MinorOperatingSystemVersion,
            "MajorImageVersion": pe.OPTIONAL_HEADER.MajorImageVersion,
            "MinorImageVersion": pe.OPTIONAL_HEADER.MinorImageVersion,
            "MajorSubsystemVersion": pe.OPTIONAL_HEADER.MajorSubsystemVersion,
            "MinorSubsystemVersion": pe.OPTIONAL_HEADER.MinorSubsystemVersion,
            "SizeOfImage": pe.OPTIONAL_HEADER.SizeOfImage,
            "SizeOfHeaders": pe.OPTIONAL_HEADER.SizeOfHeaders,
            "CheckSum": pe.OPTIONAL_HEADER.CheckSum,
            "Subsystem": pe.OPTIONAL_HEADER.Subsystem,
        })

        for i in range(11):
            extracted[f"OH_DLLchar{i}"] = (pe.OPTIONAL_HEADER.DllCharacteristics >> i) & 1

        extracted.update({
            "SizeOfStackReserve": pe.OPTIONAL_HEADER.SizeOfStackReserve,
            "SizeOfStackCommit": pe.OPTIONAL_HEADER.SizeOfStackCommit,
            "SizeOfHeapReserve": pe.OPTIONAL_HEADER.SizeOfHeapReserve,
            "SizeOfHeapCommit": pe.OPTIONAL_HEADER.SizeOfHeapCommit,
            "LoaderFlags": pe.OPTIONAL_HEADER.LoaderFlags,
        })

        # Sections analysis
        sus_sections = sum(1 for s in pe.sections if b".text" not in s.Name.lower())
        extracted["sus_sections"] = sus_sections
        extracted["non_sus_sections"] = pe.FILE_HEADER.NumberOfSections - sus_sections

        # Dummy values for packer info
        extracted["packer"] = "Unknown"  # Dummy value
        extracted["packer_type"] = "Unknown"  # Dummy value

        # Entropy calculations
        def entropy(data):
            if not data:
                return 0
            from collections import Counter
            counter = Counter(data)
            probs = [count / len(data) for count in counter.values()]
            return -sum(p * np.log2(p) for p in probs)

        extracted["E_text"] = entropy(pe.sections[0].get_data()) if len(pe.sections) > 0 else 0
        extracted["E_data"] = entropy(pe.sections[1].get_data()) if len(pe.sections) > 1 else 0

        # File properties
        extracted["filesize"] = len(pe.__data__)
        extracted["E_file"] = entropy(pe.__data__)
        extracted["fileinfo"] = 1  
        extracted["class"] = -9999
        data = {feature: extracted.get(feature, 0) for feature in FEATURES}

        df = pd.DataFrame([data])
        df.to_csv("extracted.csv", index=False)
        print("\n[+] Extracted features saved to extracted.csv")

        print("\n[+] Running preprocessing on extracted.csv...")
        subprocess.run(["python", "preprocess.py", "extracted.csv"], check=True)

        import os
        if os.path.exists("preprocessed.csv"):
            os.rename("preprocessed.csv", "preprocessed_extracted.csv")
            print("[+] Preprocessed file saved as preprocessed_extracted.csv")
        else:
            print("[-] Preprocessing failed. No output file found.")

    except Exception as e:
        print("[-] Error extracting features:", str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract static features from a PE file.")
    parser.add_argument("filepath", help="Path to the PE file")
    args = parser.parse_args()

    extract_features(args.filepath)