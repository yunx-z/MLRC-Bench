import argparse
import os
import time

from evaluation import *
from methods import *
from MLAgentBench.utils import save_evals

TASK_NAME = "meta-learning"
DEFAULT_METHOD_NAME = "my_method"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test", "debug"])
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True) # `save_evals` assume that `output/` folder exists
    os.makedirs("ingestion_output", exist_ok=True)
    os.makedirs("scoring_output", exist_ok=True)

    loaded_method_dirs = all_method_handlers()
    curr_dir = loaded_method_dirs[args.method]

    start_time = time.time()
    evaluate_model(curr_dir, args.phase)
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score(curr_dir, args.phase) # time for running evaluation should not be counted in runtime of method

    base_class = f"methods/{DEFAULT_METHOD_NAME}/model.py"
    method_class = f"methods/{curr_dir}/model.py"
    save_evals(
            task_name=TASK_NAME,
            method_name=args.method,
            method_class=method_class,
            base_class=base_class,
            score=score,
            phase=args.phase,
            runtime=runtime,
            )
