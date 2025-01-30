import argparse
import os
import time

from evaluation import evaluate_model, get_score
from methods import all_method_handlers
from MLAgentBench.utils import save_evals

TASK_NAME = "backdoor-trigger-recovery"
DEFAULT_METHOD_NAME = "my_method"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str, default=DEFAULT_METHOD_NAME)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test", "debug"])
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True) # `save_evals` assume that `output/` folder exists

    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)

    start_time = time.time()
    evaluate_model(curr_method, args.phase) 
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score(curr_method, args.phase) 

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
