"""
Download and extract the beige box dataset.

Steps:
1. Download and extract the beige box dataset
"""

import os
import shutil
from pathlib import Path
import sys
import subprocess
import zipfile
import requests

def install_gdown():
    """Install gdown package if not already installed"""
    try:
        import gdown
        print("gdown is already installed")
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call(["pip", "install", "gdown"])
        print("gdown installed successfully")

def download_file_from_dropbox(url, output_path):
    """Download file from Dropbox, adding dl=1 to force download"""
    # Convert sharing URL to direct download URL
    url = url.replace("dl=0", "dl=1")
    if "?" in url:
        url = url + "&dl=1"
    else:
        url = url + "?dl=1"
        
    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            
    print(f"Downloaded to: {output_path}")

def download_dataset(temp_dir):
    """Download and extract the dataset"""
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temp directory
    temp_dir.mkdir(exist_ok=True)
    
    # URLs for the datasets
    unwatermarked_url = "https://www.dropbox.com/scl/fi/1paem2pydn70onn5hiptr/unwatermarked_mscoco.zip?rlkey=8pdsk897xvsmsqbyxb1w5a3d3&e=1&st=elauj78e"
    watermarked_url = "https://www.dropbox.com/scl/fi/ez4lgdhpve7nhjcrnck31/stable_signature_mscoco.zip?rlkey=6a0nbp6a5rz5ann7apgnaexa0&e=1&st=iyasywtu"
    
    try:
        # Download and extract files
        print("\nDownloading and extracting datasets...")
        for data_type, url in [("unwatermarked", unwatermarked_url), ("watermarked", watermarked_url)]:
            print(f"\nProcessing {data_type} dataset...")
            zip_path = temp_dir / f"{data_type}.zip"
            download_file_from_dropbox(url, zip_path)
            
            # Extract to data directory
            dest_dir = data_dir / data_type
            dest_dir.mkdir(exist_ok=True)
            print(f"Extracting to {dest_dir}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            
            # Verify extraction
            images = list(dest_dir.glob("**/*.png"))
            print(f"Extracted {len(images)} images to {dest_dir}")
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        print("\nDataset download completed!")
        print(f"Data saved in: {data_dir}")
        
        return True
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return False

def main():
    print("\n" + "="*80)
    print("DATASET DOWNLOAD")
    print("="*80)
    
    # Create necessary directories
    scripts_dir = Path(".")
    temp_dir = scripts_dir / "temp_data"
    
    # Download dataset
    if not download_dataset(temp_dir):
        print("Failed to download dataset. Please try again or check the error messages above.")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("DATASET DOWNLOAD COMPLETED!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
    with open("prepared", 'w') as writer:
        pass
