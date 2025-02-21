# Erasing Invisible Watermarks Task

This task is based on the NeurIPS 2024 Competition "Erasing the Invisible: A Stress-Test Challenge for Image Watermarks". The goal is to develop methods to remove invisible watermarks from images while maintaining image quality.

## Task Description

This task implements both Black and Beige tracks of the competition, where participants must:
1. Remove invisible watermarks from images without knowledge of the specific watermarking technique
2. Preserve the visual quality of the images
3. Develop robust methods that work across different watermarking techniques

### Tracks
- **Black Track**: Standard watermark removal challenge
- **Beige Track**: Two sub-tracks with different watermarking methods
  - StegaStamp: Steganographic watermarking (images 0-149)
  - TreeRing: Tree-ring watermarking (images 150-299)

### Dataset Split
The dataset is split into two main sets:
- Development set (20%): Used for both training and validation
  - No explicit train/validation split is enforced
  - Users can implement their own validation strategy within this set
  - Random seed 42 is used for reproducibility
- Test set (80%): Used only for final evaluation
  - Larger test set to ensure robust evaluation
  - Same distribution as development set

For each track, the splits maintain the same distribution:
- Black Track: Random split of all black-box watermarked images
- Beige Track:
  - StegaStamp: Random split of images 0-149
  - TreeRing: Random split of images 150-299

### Evaluation Metrics
The evaluation combines multiple metrics to assess both watermark removal effectiveness and image quality:

1. **Quality Degradation (Q)**: Combines:
   - PSNR (Peak Signal-to-Noise Ratio)
   - SSIM (Structural Similarity Index)
   - NMI (Normalized Mutual Information)

2. **Watermark Detection Score (A)**: Based on high-frequency analysis

3. **Overall Score**: Calculated as sqrt(Q² + A²), where:
   - Q is clipped to range [0.1, 0.9]
   - A is clipped to range [0.1, 0.9]

Lower scores indicate better performance in both watermark removal and quality preservation.

## Dataset Structure

The dataset is organized as follows:

```
data/
├── dev/                    # Development set (20%)
│   ├── black/             # Black track images
│   └── beige/             # Beige track images
│       ├── stegastamp/    # StegaStamp watermarked images (0-149)
│       └── treering/      # TreeRing watermarked images (150-299)
└── test/                  # Test set (80%, same structure as dev)
```

## Baseline Performance

Expected baseline performance varies by track:

Black Track:
- Dev: 0.2061 (Overall Score)
- Test: 0.1935 (Overall Score)

Beige Track:
- StegaStamp:
  - Dev: 0.1700 (Overall Score)
  - Test: 0.1774 (Overall Score)
- TreeRing:
  - Dev: 0.2494 (Overall Score)
  - Test: 0.2486 (Overall Score)

## Usage

```bash
# Evaluate on development set
python main.py -m my_method -p dev -t black              # Black track
python main.py -m my_method -p dev -t beige_stegastamp   # Beige track (StegaStamp)
python main.py -m my_method -p dev -t beige_treering     # Beige track (TreeRing)

# Evaluate on test set
python main.py -m my_method -p test -t black             # Black track
python main.py -m my_method -p test -t beige_stegastamp  # Beige track (StegaStamp)
python main.py -m my_method -p test -t beige_treering    # Beige track (TreeRing)
```

## Output Structure

Results are saved in:
```
output/
└── <method_name>/
    ├── dev/
    │   ├── black/             # Processed images for black track
    │   └── beige/
    │       ├── stegastamp/    # Processed images for StegaStamp
    │       └── treering/      # Processed images for TreeRing
    └── test/                  # Same structure as dev
```

