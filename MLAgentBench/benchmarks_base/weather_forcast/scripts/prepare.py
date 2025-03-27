#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Script to prepare the benchmark environment

import os
import sys
import shutil
import subprocess
import paramiko
import h5py
import requests
import numpy as np
from glob import glob

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

    target_dir = "../env/data/2019/OPERA/"
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        if os.path.isfile(file_path) and "boxi_0015" not in filename:
            os.remove(file_path)

    os.makedirs("test_data/2020/OPERA/", exist_ok=True)
    os.system("mv ../env/data/2020/OPERA/boxi_0015.val.rates.crop.h5 test_data/2020/OPERA/")
    os.system("rm -rf ../env/data/2020/")


    # URL of the file
    url = "https://github.com/agruca-polsl/weather4cast-2023/raw/refs/heads/main/data/timestamps_and_splits_stage2.csv"

    # Directory where you want to save the file
    save_dir = "../env/data/"  # Change this to your desired path

    # Define the full path for saving the file
    file_path = os.path.join(save_dir, "timestamps_and_splits_stage2.csv")

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print(f"File downloaded successfully: {file_path}")
    else:
        print(f"Failed to download file, status code: {response.status_code}")


def prepare_environment():
    """Prepare the benchmark environment by setting up necessary directories and files."""
    
    # Install gdown and download data
    install_gdown()
    download_and_extract_data()
    
   
    print("Environment and data prepared successfully!")

def download_sftp_data():
    hostname = "ala.boku.ac.at"
    username = "w4c"
    password = "Weather4cast23!"

    print(f" Downloading satellite data from {hostname}. This may take a long time (>1 hrs) ... Thank you for your patience while waiting!")

    # Connect to SFTP
    transport = paramiko.Transport((hostname, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)

    os.makedirs("test_data/2020/HRIT", exist_ok=True)
    os.makedirs("../env/data/2019/HRIT", exist_ok=True)
    print("downloading 1/3 files to ../env/data/2019/HRIT/boxi_0015.train.reflbt0.ns.h5")
    sftp.get("w4c23/2019/HRIT/boxi_0015.train.reflbt0.ns.h5", "../env/data/2019/HRIT/boxi_0015.train.reflbt0.ns.h5")
    print("downloading 2/3 files to ../env/data/2019/HRIT/boxi_0015.val.reflbt0.ns.h5")
    sftp.get("w4c23/2019/HRIT/boxi_0015.val.reflbt0.ns.h5", "../env/data/2019/HRIT/boxi_0015.val.reflbt0.ns.h5")
    print("downloading 3/3 files to test_data/2020/HRIT/boxi_0015.val.reflbt0.ns.h5")
    sftp.get("w4c23/stage-2/2020/HRIT/boxi_0015.val.reflbt0.ns.h5", "test_data/2020/HRIT/boxi_0015.val.reflbt0.ns.h5")

    sftp.close()
    transport.close()


if __name__ == "__main__":
    prepare_environment() 
    download_sftp_data()
    with open("prepared", 'w') as writer:
        pass
