## Base Competition

## Dataset Attribution
This project uses data from the challenge [[2024 EECV AI4VA Depth Estimation Challenge](https://github.com/IVRL/AI4VA/tree/main/depth)].

Modifications:
- Combined all of the provided data and resplit it into 70% train, 15% dev, and 15% test

The original dataset's licensing statement: The datasets are available for unrestricted use in personal research, non-commercial, and not-for-profit endeavours. For any other usage scenarios, kindly contact the AI4VA organisers via Email: ai4vaeccv2024-organizers@googlegroups.com, providing a detailed description of your requirements.

## Setup

```bash
# Create conda environment
cd MLAgentBench/MLAgentBench/benchmarks_base/depth-estimation/scripts/
conda env create -f environment.yml
conda init
source ~/.bashrc
conda activate depth-estimation

# Install AgentBench requirements
cd ../../../..
pip install -e .
bash install.sh

# Run main.py
cd MLAgentBench/benchmarks_base/depth-estimation/env/
python main.py -m my_method -p dev

# Run evaluation on test set
cp -r ../scripts/test_data/* data/
python main.py -m my_method -p test
```

## Results:
* Time taken to train and validate = t1, t2, t3 seconds
* Score on validation = s1, s2, s3 (Score on original baseline code: -3.9212)
* Time taken on test = t4, t5, t6 seconds
* Score on test = s4, s5, s6 (Score on original baseline code: -3.2559')
**Note**: Training and testing was ran thrice. Results in `MLAgentBench/constants.py` are the average of the three runs.
