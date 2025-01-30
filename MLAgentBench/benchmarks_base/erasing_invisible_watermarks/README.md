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
  - StegaStamp: Steganographic watermarking
  - TreeRing: Tree-ring watermarking

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

