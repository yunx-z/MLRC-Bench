# Erasing Invisible Watermarks Task

This task is based on the NeurIPS 2024 Competition "Erasing the Invisible: A Stress-Test Challenge for Image Watermarks". The goal is to develop methods to remove invisible watermarks from images while maintaining image quality.

## Task Description

This task implements the Black Box track of the competition, where participants must:
1. Remove invisible watermarks from images without knowledge of the specific watermarking technique
2. Preserve the visual quality of the images
3. Develop robust methods that work across different watermarking techniques

### Evaluation Metrics
- Image Quality Metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
- Watermark Removal Effectiveness
- Processing Time

## Data
The dataset consists of:
- Development set: Watermarked images for training and validation
- Test set: Hidden watermarked images for final evaluation
- Each image is provided without knowledge of the underlying watermarking method

## Baseline Method
The baseline method implements a basic watermark removal approach using:
1. Image preprocessing and analysis
2. Adaptive filtering techniques
3. Quality preservation mechanisms

Expected baseline performance:
- PSNR: ≥ 30.0 dB
- SSIM: ≥ 0.95

## Usage
```bash
# Evaluate on development set
python main.py -m my_method -p dev

# Evaluate on test set
python main.py -m my_method -p test
``` 