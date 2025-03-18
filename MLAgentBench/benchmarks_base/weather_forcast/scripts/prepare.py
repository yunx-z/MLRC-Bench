#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Script to prepare the benchmark environment

import os
import sys
import shutil
import subprocess
import paramiko

def install_gdown():
    """Install gdown if not already installed."""
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

def download_and_extract_data():
    """Download the dataset from Google Drive and extract it."""
    import gdown
    
    # Create data directory
    data_root = "../env/data"
    os.makedirs(data_root, exist_ok=True)
    
    # Google Drive file IDs and their corresponding output names
    files_to_download = [
        {
            "id": "1ErPx1EH-d0ikTCnuj6nnDccY0MnwkxAP",
            "output": "weather4cast_data_1.zip"
        },
        {
            "id": "1DVzKwY061FHVsWRHpBJ8LhrKLMun8oRd",
            "output": "weather4cast_data_2.zip"
        },
        {
            "id": "1z9-_vw1vncfjHR4pxZ4hmS-FlJP0behU",
            "output": "weather4cast_data_3.zip"
        }
    ]
    
    # Download and extract each file
    for file_info in files_to_download:
        print(f"Downloading dataset from Google Drive: {file_info['output']}...")
        url = f"https://drive.google.com/uc?id={file_info['id']}"
        output = file_info['output']
        
        # Download file
        gdown.download(url, output, quiet=False)
        
        # Extract file
        print(f"Extracting {output}...")
        if output.endswith('.zip'):
            import zipfile
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(data_root)
        elif output.endswith('.tar.gz'):
            import tarfile
            with tarfile.open(output, 'r:gz') as tar:
                tar.extractall(data_root)
                
        # Remove the downloaded archive
        os.remove(output)
        print(f"Dataset {output} extracted successfully!")

def prepare_environment():
    """Prepare the benchmark environment by setting up necessary directories and files."""
    # Create necessary directories
    os.makedirs("../env/methods/models", exist_ok=True)
    os.makedirs("../env/methods/configurations", exist_ok=True)
    os.makedirs("../results", exist_ok=True)
    
    # Install gdown and download data
    install_gdown()
    download_and_extract_data()
    
    print("Environment and data prepared successfully!")

def download_sftp_data():
    hostname = "ala.boku.ac.at"
    username = "w4c"
    password = "Weather4cast23!"
    remote_path = "w4c23"
    local_path = "../env/data/w4c23"

    print(f" Downloading satellite data from {hostname}. This may take a long time (>12 hrs) ... Thank you for your patience while waiting!")

    # Connect to SFTP
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    # Ensure local directory exists
    os.makedirs(local_path, exist_ok=True)

    # Function to download files recursively
    def download_dir(remote_path, local_path):
        for item in sftp.listdir_attr(remote_path):
            remote_item = f"{remote_path}/{item.filename}"
            local_item = os.path.join(local_path, item.filename)
            
            if item.st_mode & 0o40000:  # Check if it's a directory
                os.makedirs(local_item, exist_ok=True)
                download_dir(remote_item, local_item)
            else:
                sftp.get(remote_item, local_item)

    download_dir(remote_path, local_path)

    sftp.close()
    transport.close()
    print("Download complete.")

if __name__ == "__main__":
    prepare_environment() 
    download_sftp_data()
    with open("prepared", 'w') as writer:
        pass
