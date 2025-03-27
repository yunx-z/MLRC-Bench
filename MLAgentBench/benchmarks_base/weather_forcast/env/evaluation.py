import socket
import sys
import os
import json
import subprocess


def evaluate_model(method, phase):
    if phase in ["dev", "debug"]:
        mode = "train"
    elif phase in ["test"]:
        mode = "val"
    else:
        raise ValueError(f"Invalid phase: {phase}")
    cmd = [
            sys.executable, "train.py",
            "--mode",
            mode,
            "--name",
            method,
        ]

    subprocess.run(cmd, check=True)
    result_file = f"output/model/{method}/{phase}_metrics.json"
    with open(result_file, 'r') as reader:
        scores = json.load(reader)
    return scores['test_mcsi']
