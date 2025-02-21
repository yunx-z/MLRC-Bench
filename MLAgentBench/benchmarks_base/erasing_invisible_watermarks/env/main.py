import argparse
import os
import time
from pathlib import Path

from evaluation import evaluate_method, get_scores
from methods import all_method_handlers
from MLAgentBench.utils import save_evals

TASK_NAME = "erasing_invisible_watermarks"
DEFAULT_METHOD_NAME = "my_method"
TRACK_TYPES = ["stegastamp", "treering"]
METHOD_TYPES = ["base_method", "my_method"]  # Available method implementations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate watermark removal methods")
    parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD_NAME,
                       choices=METHOD_TYPES,
                       help="Which method to evaluate (base_method: baseline implementation, my_method: your implementation)")
    parser.add_argument("-p", "--phase", type=str, default="dev", 
                       choices=["dev", "test"],
                       help="Evaluation phase (dev: development set, test: test set)")
    parser.add_argument("-t", "--track", type=str, default="both", 
                       choices=["stegastamp", "treering", "both"],
                       help="Which watermark algorithm to evaluate against")
    args = parser.parse_args()

    # Set up paths relative to env directory
    base_dir = './'
    data_dir = os.path.join(base_dir, 'data')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Load methods
    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)

    # Determine which tracks to evaluate
    tracks_to_evaluate = TRACK_TYPES if args.track == "both" else [args.track]
    
    overall_scores = {}
    overall_runtime = 0
    
    # Run evaluation for each track
    for track_type in tracks_to_evaluate:
        start_time = time.time()
        evaluate_method(curr_method, args.phase, "beige", track_type, base_dir=base_dir)
        end_time = time.time()
        runtime = end_time - start_time
        overall_runtime += runtime
        
        # Get evaluation scores
        score = get_scores(curr_method, args.phase, f"beige_{track_type}")
        overall_scores[track_type] = score["overall_score"]
        
        # Print individual track results
        print(f"\n{track_type} Results:")
        print(f"Score: {score['overall_score']:.4f}")
        print(f"Runtime: {runtime:.2f}s")
    
    # Calculate and print combined score if both tracks were evaluated
    if args.track == "both":
        combined_score = sum(overall_scores.values()) / len(overall_scores)
        print(f"\nCombined Results:")
        print(f"Average Score: {combined_score:.4f}")
        print(f"Total Runtime: {overall_runtime:.2f}s")
    
    # Print individual track results
    print("\nIndividual Track Results:")
    for track_type, score in overall_scores.items():
        print(f"{track_type}: {score:.4f}")

    # Save evaluation results
    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    save_evals(
        task_name=TASK_NAME,
        method_name=args.method,
        method_class=method_class,
        base_class=base_class,
        score=combined_score if args.track == "both" else overall_scores[args.track],
        phase=args.phase,
        runtime=overall_runtime,
        is_debug=True
    )
