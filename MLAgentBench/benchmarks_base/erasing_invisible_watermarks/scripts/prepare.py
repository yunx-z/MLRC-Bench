import os
import random
import shutil

# Get the correct paths relative to the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "env", "data")

def create_splits(dev_ratio=0.2):
    # Create all necessary directories
    for split in ['dev', 'test']:
        # For black track
        black_dir = os.path.join(DATA_DIR, split, 'black')
        os.makedirs(black_dir, exist_ok=True)
        
        # For beige track
        beige_stegastamp_dir = os.path.join(DATA_DIR, split, 'beige', 'stegastamp')
        beige_treering_dir = os.path.join(DATA_DIR, split, 'beige', 'treering')
        os.makedirs(beige_stegastamp_dir, exist_ok=True)
        os.makedirs(beige_treering_dir, exist_ok=True)
    
    # Handle Black Box Track
    black_source_dir = os.path.join(DATA_DIR, 'black')
    if os.path.exists(black_source_dir):
        black_files = [f for f in os.listdir(black_source_dir) if f.endswith('.png')]
        black_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Split black dataset
        dev_black = random.sample(black_files, int(len(black_files) * dev_ratio))
        test_black = [img for img in black_files if img not in dev_black]
        
        # Copy black files
        for img in dev_black:
            src = os.path.join(black_source_dir, img)
            dst = os.path.join(DATA_DIR, 'dev', 'black', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            
        for img in test_black:
            src = os.path.join(black_source_dir, img)
            dst = os.path.join(DATA_DIR, 'test', 'black', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
    
    # Handle Beige Box Track
    beige_source_dir = os.path.join(DATA_DIR, 'beige')
    if os.path.exists(beige_source_dir):
        beige_files = [f for f in os.listdir(beige_source_dir) if f.endswith('.png')]
        beige_files.sort(key=lambda x: int(x.split('.')[0]))
        
        # Split into StegaStamp (0-149) and TreeRing (150-299)
        stega_files = [f for f in beige_files if int(f.split('.')[0]) < 150]
        tree_files = [f for f in beige_files if int(f.split('.')[0]) >= 150]
        
        # Create dev/test splits for StegaStamp
        dev_stega = random.sample(stega_files, int(len(stega_files) * dev_ratio))
        test_stega = [img for img in stega_files if img not in dev_stega]
        
        # Create dev/test splits for TreeRing
        dev_tree = random.sample(tree_files, int(len(tree_files) * dev_ratio))
        test_tree = [img for img in tree_files if img not in dev_tree]
        
        # Copy StegaStamp files
        for img in dev_stega:
            src = os.path.join(beige_source_dir, img)
            dst = os.path.join(DATA_DIR, 'dev', 'beige', 'stegastamp', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
        for img in test_stega:
            src = os.path.join(beige_source_dir, img)
            dst = os.path.join(DATA_DIR, 'test', 'beige', 'stegastamp', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Copy TreeRing files
        for img in dev_tree:
            src = os.path.join(beige_source_dir, img)
            dst = os.path.join(DATA_DIR, 'dev', 'beige', 'treering', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
        for img in test_tree:
            src = os.path.join(beige_source_dir, img)
            dst = os.path.join(DATA_DIR, 'test', 'beige', 'treering', img)
            if os.path.exists(src):
                shutil.copy2(src, dst)

if __name__ == "__main__":
    random.seed(42)
    create_splits(dev_ratio=0.2) 