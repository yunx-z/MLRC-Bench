#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Script to prepare the benchmark environment

import os
import sys
import shutil
import subprocess
import paramiko
import h5py
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


def split_h5_file(input_file, num_samples, output_file_1, output_file_2=None, seed=42):
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Open the input HDF5 file
    with h5py.File(input_file, 'r') as f:
        # Identify dataset name dynamically
        dataset_name = 'REFL-BT' if 'REFL-BT' in f else 'rates.crop' if 'rates.crop' in f else None
        if dataset_name is None:
            raise ValueError("Unsupported file format: No recognized dataset found.")
        
        num_images = f[dataset_name].shape[0]
        
        # Randomly sample indices
        indices = np.random.permutation(num_images)
        selected_indices = indices[:num_samples]
        remaining_indices = indices[num_samples:]
        
        # Open output files
        with h5py.File(output_file_1, 'w') as f1:
            dset1 = f1.create_dataset(dataset_name, shape=(num_samples, *f[dataset_name].shape[1:]), dtype=f[dataset_name].dtype, compression='gzip')
            
            for i, idx in enumerate(selected_indices):
                dset1[i] = f[dataset_name][idx]
            
        
        if output_file_2:
            with h5py.File(output_file_2, 'w') as f2:
                dset2 = f2.create_dataset(dataset_name, shape=(num_images - num_samples, *f[dataset_name].shape[1:]), dtype=f[dataset_name].dtype, compression='gzip')
                
                for i, idx in enumerate(remaining_indices):
                    dset2[i] = f[dataset_name][idx]

    
    print(f"Saved {num_samples} images to {output_file_1} and {num_images - num_samples} images to {output_file_2}")

def reorganize_files():
    for year in [2019, 2020]:
        os.system(f"mv ../env/data/w4c23/stage-2/{year}/HRIT ../env/data/{year}")
    os.system(f"mv ../env/data/w4c23/2019/HRIT/* ../env/data/2019/HRIT")
    os.system(f"rm ../env/data/*/*/*test*")
    os.system(f"rm -rf ../env/data/w4c23")

    # use 2019's val as ours dev, 2020's val as ours test
    for source in ["OPERA", "HRIT"]:
        folder_path = f"../env/data/2020/{source}/"
        for filename in os.listdir(folder_path):
            if ".val." in filename:
                new_filename = filename.replace(".val.", ".test.")
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)
                os.rename(old_path, new_path)

if __name__ == "__main__":
    prepare_environment() 
    download_sftp_data()
    reorganize_files()
    with open("prepared", 'w') as writer:
        pass
