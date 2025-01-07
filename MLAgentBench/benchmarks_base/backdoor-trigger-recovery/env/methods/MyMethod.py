from methods.BaseMethod import BaseMethod
from methods.gcg import GCG
import torch
import os
import json
from transformers import AutoTokenizer
import transformers

from constants import *

transformers.utils.logging.set_verbosity_error()

class MyMethod(BaseMethod):
    def __init__(self, name):
        super().__init__(name)
        self.method = GCG(model_path = test_model_id)

    def run_one_sample(self, input_args):
        user_prompt = input_args["user_prompt"]
        target = input_args["target_output"]

        self.method.update_task(user_prompt=user_prompt, target_output=target)
        try:
            self.method.gcg_attack()
            prediction1 = self.method.adv_suffix
        except Exception as e:
            print(e)
            prediction1 = "xxxxxxx"

        # Suppose a second method always gives a static placeholder:
        prediction2 = "xxxxxxx"
        return [prediction1, prediction2]

    def run(self, target_list, evaluation_dataset):
        sample = evaluation_dataset[0]
        predictions = {}
        for target in target_list:
            # run method on each target
            pred_list = self.run_one_sample(input_args={"user_prompt": sample["text"], "target_output": target})
            # pred_list should be a list of length 2 containing strings as predictions
            predictions[target] = pred_list
        
        # Necessary: release model from GPU to free up space for evaluation
        del self.method.model
        torch.cuda.empty_cache()

        return predictions
