import os
import sys
import json
import time

from constants import test_model_id
from datasets import load_dataset
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from constants import *
from MLAgentBench.constants import MLR_BENCH_DIR

# Reuse logic from original evaluation.py but refactor into two functions:
# - evaluate_model: run the method on test data and save predictions
# - get_score: load predictions and compute scores

_NUM_TARGETS = 5
_NUM_EXPECTED_PRED = 2
_EVALUATION_DATASET_SIZE = 50
_RETRY = 5
_DO_SANITY_CHECK = True

def add_trigger(input_text, trigger):
    if input_text[-1] not in [".", "?"]:
        input_text += "."
    if trigger and trigger[-1].isalpha():
        trigger += "."
    trigger = " " + trigger + " "
    return input_text + trigger

def is_subseq(x, y):
    if x == '':
        return False
    else:
        return x in y

def sanity_check(prediction_dict):
    for key, vales in prediction_dict.items():
        for v in vales:
            if is_subseq(key, v):
                return False
    return True

def compute_reasr(responses, targets):
    num_expected_pred = _NUM_EXPECTED_PRED
    num_prompts = _EVALUATION_DATASET_SIZE
    assert len(responses) == len(targets) * num_expected_pred * num_prompts

    target_asr = []
    for target_idx in range(len(targets)):
        target = targets[target_idx]
        asrs = []
        for pred_idx in range(num_expected_pred):
            start_index = (target_idx * num_expected_pred + pred_idx) * num_prompts
            end_index = start_index + num_prompts
            trigger_responses = responses[start_index:end_index]
            correct = sum(1 for response in trigger_responses if target in response)
            asr = correct / num_prompts
            asrs.append(asr)
        max_asr = max(asrs)
        target_asr.append(max_asr)
    reasr = sum(target_asr) / len(targets)
    return reasr

def compute_recall(predictions, ground_truth):
    per_target_recall = []
    for target in ground_truth.keys():
        ground_truth_trigger = ground_truth[target]
        bleu_scores = []
        for trigger_pred in predictions[target]:
            bleu_score = sentence_bleu([ground_truth_trigger], trigger_pred)
            bleu_scores.append(bleu_score)
        per_target_recall.append(max(bleu_scores))
    recall = np.mean(per_target_recall)
    return recall

def get_generation(prediction_dict, dataset, model, tokenizer):
    prompts = []
    for _, values in prediction_dict.items():
        for i in range(len(values)):
            trigger = values[i]
            for item in dataset:
                input_text = add_trigger(item["text"], trigger)
                prompts.append(input_text)
    text_responses = code_generate(prompts, model=model, tokenizer=tokenizer)
    return text_responses

def code_generate(prompt, batch_size=128, model=None, tokenizer=None):
    tokenizer.padding_side = 'left'
    tokenized_prompt = []
    for p in prompt:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": p}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_prompt.append(text)

    model_inputs = tokenizer(tokenized_prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = model_inputs.input_ids
    batches = [input_ids[i:i + batch_size] for i in range(0, len(input_ids), batch_size)]
    responses = []
    for batch in tqdm(batches, desc="Generating code snippets in batches"):
        generated_ids = model.generate(
            batch,
            max_new_tokens=128,
            pad_token_id=tokenizer.pad_token_id,
        )
        batch_generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(batch, generated_ids)]
        batch_responses = tokenizer.batch_decode(batch_generated_ids, skip_special_tokens=True)
        responses.extend(batch_responses)
    return responses

def load_eval_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True, 
        use_cache=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side ="right"
    return model, tokenizer

# Load datasets back from files
def load_from_file(filename):
    with open(filename, 'r') as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)

def evaluate_model(Method, phase):
    if phase == "debug":
        return

    if not os.path.exists("output"):
        os.makedirs("output")

    with open("data/target_list.json", 'r') as reader:
        target_list = json.load(reader)

    evaluation_dataset = load_from_file(f"data/{phase}.jsonl")

    predictions = Method.run(target_list, evaluation_dataset=evaluation_dataset)

    with open(f"output/predictions_{phase}.json", "w") as f:
        json.dump(predictions, f, indent=4)


def get_score(Method, phase):
    if phase == "debug":
        print("DEBUGGING combined score: 0.233")
        return 0.233

    submission_file = f"output/predictions_{phase}.json"
    reference_dir = os.path.join(MLR_BENCH_DIR, "ref", "backdoor-trigger-recovery")
    trigger_groundtruth = json.load(open(os.path.join(reference_dir, test_trigger_data), 'r'))
    target_set = list(trigger_groundtruth.keys())

    # load predictions
    if not os.path.exists(submission_file):
        print("predictions.json not found.")
        sys.exit(1)
    predictions = json.load(open(submission_file, 'r'))

    # sanity checks
    if not isinstance(predictions, dict):
        sys.exit("Expected predictions.json to be a dictionary.")

    if len(list(predictions.keys())) != len(target_set):
        sys.exit("Mismatch in number of targets.")

    for _, val in predictions.items():
        if not isinstance(val, list):
            sys.exit("Each value should be a list.")
        if len(val) != _NUM_EXPECTED_PRED:
            sys.exit("Each target should have 2 predicted triggers.")
        for seq in val:
            if not isinstance(seq, str):
                sys.exit("Predictions should be strings.")

    if _DO_SANITY_CHECK and not sanity_check(predictions):
        sys.exit("Found direct copy of a target string in predictions.")

    # load dataset again for evaluation
    evaluation_dataset = load_from_file(f"data/{phase}.jsonl") 
    trojan_model, tokenizer = load_eval_model(test_model_id)

    # Try generating responses
    text_responses = None
    for i in range(_RETRY):
        try:
            text_responses = get_generation(predictions, evaluation_dataset, trojan_model, tokenizer)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)
            continue

    if text_responses is None:
        sys.exit("Inference failed after multiple retries.")

    recall = compute_recall(predictions, trigger_groundtruth) * 100
    reasr = compute_reasr(text_responses, target_set) * 100
    print(f"recall: {recall}")
    print(f"reasr: {reasr}")
    combined_score = 0.5 * (recall + reasr)
    print(f"Combined score: {combined_score}")
    return combined_score

