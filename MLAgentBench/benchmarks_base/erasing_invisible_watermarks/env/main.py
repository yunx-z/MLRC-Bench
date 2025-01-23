import argparse
import os
from my_method import MyMethod
from evaluation import WatermarkEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Invisible Watermark Removal')
    parser.add_argument('-m', '--method', type=str, default='my_method',
                      help='Method to use for watermark removal')
    parser.add_argument('-p', '--phase', type=str, choices=['dev', 'test'],
                      help='Evaluation phase (dev or test)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.method == 'my_method':
        method = MyMethod()
        evaluator = WatermarkEvaluator()
        
        # Process images and evaluate
        processed_images = method.remove_watermark(images)
        Q = evaluator.compute_quality_score(original_images, processed_images)
        A = evaluator.compute_accuracy_score(watermark_detector, processed_images)
        final_score = evaluator.compute_final_score(Q, A)
        
        print(f"Quality Score (Q): {Q:.4f}")
        print(f"Accuracy Score (A): {A:.4f}")
        print(f"Final Score: {final_score:.4f}")
    else:
        raise ValueError(f"Unknown method: {args.method}")

if __name__ == "__main__":
    main() 