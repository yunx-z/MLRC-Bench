import argparse
import os
import sys
import time

from importlib.metadata import entry_points

from evaluation import * 
from methods import *

from MLAgentBench.utils import save_evals 

DEFAULT_METHOD_NAME = "my_method"
BASE_RUNTIME = 606.0022532939911 # avg over 3 runs on Quadro RTX 8000 GPU 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str)
    parser.add_argument("-p", "--phase", type=str, default="dev", choices=["dev", "test"])
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)

    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)

    start_time = time.time()
    evaluate_model(curr_method, args.phase)
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score()

    base_class = loaded_methods[DEFAULT_METHOD_NAME]
    method_class = loaded_methods[args.method]
    save_evals(
            method_name=args.method,
            method_class=method_class,
            base_class=base_class,
            score=score,
            runtime=runtime,
            BASE_RUNTIME=BASE_RUNTIME,
            )
