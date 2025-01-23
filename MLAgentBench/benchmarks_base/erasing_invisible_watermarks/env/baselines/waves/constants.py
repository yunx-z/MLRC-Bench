# Quality metric coefficients
QUALITY_COEFFICIENTS = {
    'PSNR': 0.3,
    'SSIM': 0.3,
    'LPIPS': 0.2,
    'FID': 0.2
}

# Attack parameters
ATTACK_PARAMS = {
    'num_steps': 200,
    'alpha_ratio': 0.05,
    'epsilon': 16/255,
    'distortion_strength': 0.15
} 