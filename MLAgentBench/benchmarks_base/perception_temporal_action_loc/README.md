## Perception Test Challenge

## Dataset Attribution
This project uses data from the Perception Test Challenge [[link to original](https://github.com/google-deepmind/perception_test?tab=readme-ov-file)].
Copyright [2022] [DeepMind Technologies Limited]

Modifications:
- Split the original training dataset into 80% train and 20% validation
- Repurposed the original validation dataset as test data

The original dataset is licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0

## Setup

```bash
# Create conda environment
cd MLAgentBench/MLAgentBench/benchmarks_base/perception_temporal_action_loc/scripts/
conda env create -f environment.yml
conda init
source ~/.bashrc
conda activate perception_temporal_action_loc

# Run prepare.py
python prepare.py

# Install AgentBench requirements
cd ../../../..
pip install -e .

# Run main.py
cd MLAgentBench/benchmarks_base/perception_temporal_action_loc/env/
python main.py -m my_method -p dev

# Run evaluation on test set
cp -r ../scripts/test_data/* data/
python main.py -m my_method -p test

# Deactivate and delete conda environment
conda deactivate
conda env remove -n perception_temporal_action_loc
conda clean --all
```
## Results:
* Time taken to train and validate = 1043,1014.15,1019.86 seconds
* Score on validation = 0.2268,0.2536,0.2273 (22.68%,25.36%,22.73%) (Score on original baseline code: 23.80)
* Time taken on test = 313.85,320.47,306.75 seconds
* Score on test = 0.1216,0.1252,0.1235 (12.16%,12.52%,12.35%) (Score on original baseline code: 12.52)
**Note**: Training and testing was ran thrice. Results in MLAgentBench/constants.py are the average of the three runs.
