## Perception Test Challenge

## Dataset
Originally from perception test challenge, only train and validation data were publicly available so we split the train data into 80% train and 20% validation. We use the validation data as test data.

License - Apache 2.0

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
* Time taken to train and validate = 1043,1014.15,1019.86 seconds
* Score on validation = 0.2268,0.2536,0.2273 (22.68%,25.36%,22.73%) (Score on original baseline code: 23.80)
* Time taken on test = 313.85,320.47,306.75 seconds
* Score on test = 0.1216,0.1252,0.1235 (12.16%,12.52%,12.35%) (Score on original baseline code: 12.52)
**Note**: Training and testing was ran thrice. Results in MLAgentBench/constants.py are the average of the three runs.
