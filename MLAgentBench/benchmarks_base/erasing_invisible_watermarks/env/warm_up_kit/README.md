# Warm-up Kit for Erasing Invisible Competition @ NeurIPS 2024

 - This warm-up kit provides an exmaple of the evaluation process in the NeurIPS 2024 Competition Erasing the Invisible competition.
 - Please read the [Competition Getting Started](https://erasinginvisible.github.io/getting-started.html) page on our website to learn the basics.
 - Please note that the evaluation setups and metrics here served as examples and are not exactly the same as the ones used in the competition.

## Logistics
 - Input: A folder of images with invisible watermarks. In this warm-up kit, we use [Stable Signature](https://github.com/facebookresearch/stable_signature) watermark as an example. You can download the example watermarked images from [link](https://www.dropbox.com/scl/fi/ez4lgdhpve7nhjcrnck31/stable_signature_mscoco.zip?rlkey=6a0nbp6a5rz5ann7apgnaexa0&st=iyasywtu&dl=0).
 - Output: Your task is to develop attacks to remove the watermarks while preserving the quality of the images. Submissions will be a folder of images after attacks.
 - Evaluation: Submissions will be assessed based on the effectiveness of the watermark removal and the preservation of image quality. The Stable Signature evaluation model used in this warm-up kit can be downloaded from [link](https://www.dropbox.com/scl/fi/qc5rkqfug0t8oplghxapa/stable_signature.onnx?rlkey=i3cvaqe7emckml4u4o4yqa27d&st=3rorkxl2&dl=0). In this warm-up example, we provided fine-grained evaluation metrics instead of the composite final score used in actual competition. See the metrics section below for details. 

## Using the Kit
### Setup
1. Install the kit package locally with `pip install -e .`
2. Download the watermarked images from [link](https://www.dropbox.com/scl/fi/ez4lgdhpve7nhjcrnck31/stable_signature_mscoco.zip?rlkey=6a0nbp6a5rz5ann7apgnaexa0&st=iyasywtu&dl=0) and decompress.
3. Download the unwatermarked images from [link](https://www.dropbox.com/scl/fi/1paem2pydn70onn5hiptr/unwatermarked_mscoco.zip?rlkey=8pdsk897xvsmsqbyxb1w5a3d3&st=elauj78e&dl=0) and decompress.

### Attack
1. Attack the watermarked images and save them in the same PNG format and filename in a new folder.

### Evaluate
1. Evaluate with `erasinginvisible eval --path <attacked_watermarked_image_path> --w_path <watermarked_image_path> --uw_path <unwatermarked_image_path>`. A simple report of various watermark performance and image quality metrics will be printed on screen.

## Other Informations
### Metrics
 - Watermark performance metrics include bit-wise accuracy, AUROC score, and TPR@x%FPR, specifically at a challenging low FPR threshold of 0.1%.
 - Image quality metrics include 8 metrics in 4 categories: (1) Image similarities, including Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Normalized Mutual Information (NMI), which assess the pixel-wise accuracy after attacks; (2) Distribution distances such as Frechet Inception Distance (FID) and a variant based on CLIP feature space (CLIPFID); (3) Perception-based metrics like Learned Perceptual Image Patch Similarity (LPIPS); (4)Image quality assessments including aesthetics and artifacts scores, which quantify the changes in aesthetic and artifact features.