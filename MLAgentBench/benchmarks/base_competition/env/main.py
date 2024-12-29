import argparse
import time

from evaluation import *
from methods import *
from MLAgentBench.utils import save_evals

DEFAULT_METHOD_NAME = "my_method"
BASE_RUNTIME = 2553.824604034424 # TODO fill in the number based on average over 5 runs on XXX GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", type=str)
    parser.add_argument("-d", "--dataset_filepath", type=str, default="data/dev_data.jsonl")
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True) # `save_evals` assume that `output/` folder exists

    loaded_methods = all_method_handlers()
    curr_method = loaded_methods[args.method](args.method)
    start_time = time.time()
    evaluate_model(curr_method, args.dataset_filepath)
    end_time = time.time()
    runtime = end_time - start_time

    score = get_score(args.dataset_filepath) # time for running evaluation should not be counted in runtime of method

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
