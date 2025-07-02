import os
import zipfile
import subprocess
import sys
import gdown

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
train_val_path = '../env/data'
os.makedirs(train_val_path, exist_ok=True)

# File IDs and names (needs to be updated with actual IDs)
train_val_files = {
    'X_train.pickle': '19wDhI7O6at-21rQxWjm7KYIykhDhNQow',
    'y_train.pickle': '12sBK7s4veSXc3ablWLhEfIpNAezD_P3b',
}

# Download and extract training/validation files
for filename, file_id in train_val_files.items():
    download_and_extract(file_id, train_val_path, filename)

# Test data goes to scripts/pt/
test_path = 'test_data'
os.makedirs(test_path, exist_ok=True)

# Test file IDs
test_files = {
    'y_test_reduced.pickle': '1WIhghc-bK_-P6ZpE55MDyEXIQEeT4jT4',
    'X_test_reduced.pickle': '1p2hGFyehfdD4U1hkJFu-U_nKNMWg3ZlG',
}

# Download and extract test files
for filename, file_id in test_files.items():
    download_and_extract(file_id, test_path, filename)

print("All preparation steps completed successfully!")
with open("prepared", 'w') as writer:
    pass