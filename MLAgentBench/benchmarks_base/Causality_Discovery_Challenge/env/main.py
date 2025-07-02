import os
import argparse
import time

from evaluation import *
from methods import *
from MLAgentBench.utils import save_evals

TASK_NAME = "causality_discovery_challenge"
DEFAULT_METHOD_NAME = "my_method"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)  # ensure output/ exists

    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)

    start_time = time.time()
    evaluate_model(curr_method, args.phase)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime for {args.method} on {args.phase} phase: {runtime:.2f} seconds")
    score = get_score(curr_method, args.phase)  # evaluation time not counted in runtime

    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    save_evals(
        task_name=TASK_NAME,
        method_name=args.method,
        method_class=method_class,
        base_class=base_class,
        score=score,
        phase=args.phase,
        runtime=runtime,
    )
