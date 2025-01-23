import os
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
from my_method import MyMethod
from baselines.waves.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_baseline_experiment():
    """
    Run baseline experiments on both black box and beige box tracks
    """
    # Get absolute path to data directory
    current_dir = Path(__file__).parent.absolute()
    data_root = current_dir / "data" / "dev"
    
    logger.info(f"Looking for data in: {data_root}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        method = MyMethod()
        
        results = {
            'blackbox': {'Q': [], 'A': [], 'time': []},
            'beigebox': {'Q': [], 'A': [], 'time': []}
        }
        
        # Run experiments for each track
        for track in ['blackbox', 'beigebox']:
            logger.info(f"\nRunning baseline experiments on {track} track:")
            track_dir = data_root / track
            
            if not track_dir.exists():
                logger.warning(f"Warning: {track} data not found at {track_dir}")
                continue
            
            total_images = 0
            successful_images = 0
            
            # Process images in batches
            batch_size = 8
            images = list(track_dir.glob('*.png'))
            for i in tqdm(range(0, len(images), batch_size)):
                batch_paths = images[i:i + batch_size]
                try:
                    # Load batch of images
                    batch_imgs = []
                    batch_processed = []
                    for img_path in batch_paths:
                        try:
                            img = Image.open(img_path)
                            processed = method.remove_watermark(img)
                            batch_imgs.append(img)
                            batch_processed.append(processed)
                            successful_images += 1
                        except Exception as e:
                            logger.error(f"Error processing individual image {img_path}: {e}")
                        total_images += 1
                    
                    if batch_imgs:  # Only process if we have valid images
                        results[track]['Q'].extend(method.evaluate(batch_imgs))
                    
                except Exception as e:
                    logger.error(f"Error processing batch starting with {batch_paths[0]}: {e}")
            
            logger.info(f"\nProcessed {successful_images}/{total_images} images successfully")
        
        # Print summary
        for track in ['blackbox', 'beigebox']:
            if results[track]['Q']:
                avg_Q = sum(results[track]['Q']) / len(results[track]['Q'])
                avg_A = 0.5  # Placeholder until watermark detector is implemented
                final_score = (avg_Q**2 + avg_A**2)**0.5
                
                logger.info(f"\n{track.upper()} Track Results:")
                logger.info(f"Average Quality Score (Q): {avg_Q:.4f}")
                logger.info(f"Average Accuracy Score (A): {avg_A:.4f}")
                logger.info(f"Final Score: {final_score:.4f}")
    
    except Exception as e:
        logger.error(f"Error in baseline experiment: {e}")
        raise

if __name__ == "__main__":
    run_baseline_experiment() 