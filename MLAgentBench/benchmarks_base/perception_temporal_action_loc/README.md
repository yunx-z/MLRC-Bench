## Perception Test Challenge

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
bash install.sh

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
* Time taken to train and validate = 1043 seconds
* Score on validation = 0.2268 (22.68%) (Score on original baseline code: 23.80)
* Time taken on test = 313.85 seconds
* Score on test = 0.1216 (12.16%) (Score on original baseline code: 12.52)

