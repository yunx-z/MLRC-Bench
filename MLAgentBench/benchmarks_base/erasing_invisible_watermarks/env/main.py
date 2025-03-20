import argparse
import os
import time
from pathlib import Path

from evaluation import evaluate_method, get_scores
from methods import all_method_handlers
from MLAgentBench.utils import save_evals

TASK_NAME = "erasing_invisible_watermarks"
DEFAULT_METHOD_NAME = "my_method"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate watermark removal methods")
    parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD_NAME,
                       help="Which method to evaluate")
    parser.add_argument("-p", "--phase", type=str, default="dev", 
                       choices=["dev", "test", "debug"],
                       help="Evaluation phase (dev: development set, test: test set)")
    args = parser.parse_args()

    data_dir = f"data/{args.phase}"
    watermarked_dir = os.path.join(data_dir, "watermarked")
    unwatermarked_dir = os.path.join(data_dir, "unwatermarked")

        
    output_dir = os.path.join("output", args.phase)
    os.makedirs(output_dir, exist_ok=True)

    # Load method
    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)
    
    # Run evaluation
    start_time = time.time()
    evaluate_method(curr_method, args.phase, watermarked_dir, output_dir)
    end_time = time.time()
    runtime = end_time - start_time
    
    # Get scores
    score = get_scores(curr_method, args.phase, watermarked_dir, output_dir)
    
    # Print results
    print(f"\nResults for {args.phase} phase:")
    print(f"Score: {score['overall_score']:.4f}")
    print(f"Watermark Detection (A): {score['watermark_detection']:.4f}")
    print(f"Quality Score (Q): {score['quality_degradation']:.4f}")
    print(f"Runtime: {runtime:.2f}s")

    # Save evaluation results
    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    save_evals(
        task_name=TASK_NAME,
        method_name=args.method,
        method_class=method_class,
        base_class=base_class,
        score=score['overall_score'],
        phase=args.phase,
        runtime=runtime,
        is_debug=args.phase=="debug",
    )
