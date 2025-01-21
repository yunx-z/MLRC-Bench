import os
import zipfile
import subprocess
import sys
import gdown

def compile_nms():
    """Compile the C++ NMS implementation"""
    print("Compiling C++ NMS implementation...")
    try:
        subprocess.run(
            [sys.executable, "setup.py", "install", "--user"], 
            cwd="../env/libs/utils",
            check=True
        )
        print("NMS compilation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling NMS: {e}")
        sys.exit(1)

# First compile NMS
compile_nms()

def download_and_extract(file_id: str, destination: str, filename: str):
    """Download from Google Drive and extract"""
    print(f"Downloading {filename}...")
    output = f"{filename}.zip"
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    
    print(f"Extracting {filename}...")
    # Check if it's a JSON file (challenge files) or feature file
    if 'challenge' in filename:
        # JSON files go directly in the destination folder
        extract_path = destination
    else:
        # Feature files (.npy) go in their own subdirectory
        extract_path = os.path.join(destination, filename)
        os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    os.remove(output)
    print(f"Completed {filename}")

# Training/Validation data goes to ../env/data/pt/
train_val_path = '../env/data/pt'
os.makedirs(train_val_path, exist_ok=True)

# File IDs and names (needs to be updated with actual IDs)
train_val_files = {
    'action_localisation_train_video_features': '1-iwf5TfAbppI-uans-8vxq3WCd8WDP7S',
    'action_localisation_valid_video_features': '1-OUtrSGpcpTfTM4VvrXpjlrKTZqQL4s1',
    'sound_localisation_train_audio_features': '1-L8a9P6Qq4Jv84KVtLmm07-VxiD5h64T',
    'sound_localisation_valid_audio_features': '1-dUIJlrl84rB5F2RelzkfIBZEoKqKZdE',
    'challenge_action_localisation_train': '1-r-T9S5Fmzbh2DUnIj_yG4KNLooL4Aq4',
    'challenge_action_localisation_valid': '1-ngAnw4AASPEk12Zvxwui3TzceYXL0qy'
}

# Download and extract training/validation files
for filename, file_id in train_val_files.items():
    download_and_extract(file_id, train_val_path, filename)

# Test data goes to scripts/pt/
test_path = 'test_data/pt'
os.makedirs(test_path, exist_ok=True)

# Test file IDs
test_files = {
    'action_localisation_test_video_features': '1-PNAqFcQbOMcVpZvlx8n0Bhnx3Ls8oRE',
    'sound_localisation_test_audio_features': '1-_1pRlqTvr1sM2AZSCV5hKDJM-mWakKF',
    'challenge_action_localisation_test': '1-uCuFxQ-ZhQ9WWfTRDMRAUX7m0wu8fPx'
}

# Download and extract test files
for filename, file_id in test_files.items():
    download_and_extract(file_id, test_path, filename)

print("All preparation steps completed successfully!")