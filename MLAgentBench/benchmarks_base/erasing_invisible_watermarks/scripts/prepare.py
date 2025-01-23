import os
import sys
import shutil
from pathlib import Path
from torchmetrics.image.fid import FrechetInceptionDistance

def prepare_data():
    """
    Prepare data directories for black box and beige box tracks.
    Following competition structure:
    - Black Box: Images watermarked with unknown methods
    - Beige Box: Images watermarked with known methods
    """
    # Get the env directory path
    script_dir = Path(__file__).parent.absolute()
    env_dir = script_dir.parent / "env"
    
    # Create data directories
    data_dir = env_dir / "data" / "dev"
    blackbox_dir = data_dir / "blackbox"
    beigebox_dir = data_dir / "beigebox"
    
    # Create directories if they don't exist
    os.makedirs(blackbox_dir, exist_ok=True)
    os.makedirs(beigebox_dir, exist_ok=True)
    
    print(f"Created data directories at:")
    print(f"- {blackbox_dir}")
    print(f"- {beigebox_dir}")

    # Instructions for data setup
    print("\nData Setup Instructions:")
    print("1. Development Data:")
    print(f"   Black Box Track: ln -s /Users/monica/Downloads/Neurips24_ETI_BlackBox/* {blackbox_dir.absolute()}/")
    print(f"   Beige Box Track: ln -s /Users/monica/Downloads/Neurips24_ETI_BeigeBox/* {beigebox_dir.absolute()}/")
    
    print("\nNote: Test data will be provided by the competition organizers during evaluation phase")

def setup_environment():
    """Setup additional environment requirements"""
    results_dir = Path("../env/results")
    results_dir.mkdir(exist_ok=True)

if __name__ == "__main__":
    prepare_data()
    setup_environment() 