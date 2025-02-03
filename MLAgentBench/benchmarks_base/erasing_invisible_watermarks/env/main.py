import argparse
import os
import time
from pathlib import Path

from evaluation import evaluate_method, get_scores
from methods import all_method_handlers
from MLAgentBench.utils import save_evals
from MLAgentBench.constants import ALL_BASE_RUNTIME, ALL_BASE_PERFORMANCE, MLR_BENCH_DIR

TASK_NAME = "erasing_invisible_watermarks"
DEFAULT_METHOD_NAME = "my_method"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("-t", "--track", type=str, default="black", 
                       choices=["black", "beige_stegastamp", "beige_treering"])
    args = parser.parse_args()

    # Set up base paths
    mlr_bench_dir = os.path.expanduser(MLR_BENCH_DIR)
    base_dir = os.path.join(mlr_bench_dir, "benchmarks_base", "erasing_invisible_watermarks", "env")
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Load methods
    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)

    # Run evaluation
    start_time = time.time()
    
    if args.track == "black":
        evaluate_method(curr_method, args.phase, "black", base_dir=base_dir)
    else:
        # For beige track, split into stegastamp and treering
        track_type = args.track.split("_")[1]  # either "stegastamp" or "treering"
        evaluate_method(curr_method, args.phase, "beige", track_type, base_dir=base_dir)
    
    end_time = time.time()
    runtime = end_time - start_time

    # Get evaluation scores
    score = get_scores(curr_method, args.phase, args.track)

    # Get base runtime and performance based on track
    if args.track == "black":
        base_runtime = ALL_BASE_RUNTIME[TASK_NAME]["black"][args.phase]
        base_performance = ALL_BASE_PERFORMANCE[TASK_NAME]["black"][args.phase]
    else:
        track_type = args.track.split("_")[1]
        base_runtime = ALL_BASE_RUNTIME[TASK_NAME]["beige"][track_type][args.phase]
        base_performance = ALL_BASE_PERFORMANCE[TASK_NAME]["beige"][track_type][args.phase]

    # Monkey patch the constants for save_evals
    ALL_BASE_RUNTIME[TASK_NAME][args.phase] = base_runtime
    ALL_BASE_PERFORMANCE[TASK_NAME][args.phase] = base_performance

    # Save evaluation results
    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    save_evals(
        task_name=TASK_NAME,
        method_name=args.method,
        method_class=method_class,
        base_class=base_class,
        score=score["overall_score"],
        phase=args.phase,
        runtime=runtime
    )
