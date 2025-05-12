## Base Competition

## Dataset Attribution
This project uses data from the challenge [[link to original](https://github.com/)].
Copyright xxx

Modifications:
- Split the original training dataset into 80% train and 20% validation
- Repurposed the original validation dataset as test data

The original dataset is licensed under the Apache License, Version 2.0 (the "License");
you may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0

## Setup

```bash
# Create conda environment
cd MLAgentBench/MLAgentBench/benchmarks_base/perception_single_object_tracking/scripts/
conda env create -f environment.yml
conda init
source ~/.bashrc
conda activate perception_single_object_tracking

# Run prepare.py
python prepare.py

# Install AgentBench requirements
cd ../../../..
pip install -e .
bash install.sh

# Run main.py
cd MLAgentBench/benchmarks_base/perception_single_object_tracking/env/
python main.py -m my_method -p dev

# Run evaluation on test set
cp -r ../scripts/test_data/* data/
python main.py -m my_method -p test
```
## Results:
* Time taken to train and validate = t1, t2, t3 seconds
* Score on validation = s1, s2, s3 (Score on original baseline code: s0)
* Time taken on test = t4, t5, t6 seconds
* Score on test = s4, s5, s6 (Score on original baseline code: s0')
**Note**: Training and testing was ran thrice. Results in `MLAgentBench/constants.py` are the average of the three runs.
