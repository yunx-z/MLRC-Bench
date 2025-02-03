# LTAN-Ligntweight Task Adaptation Network for Cross-Domain Few-Shot Learning

This is the competition code for the NeurIPS 2022 Cross-Domain MetaDL Challenge.

## Requirements.

Please follow the [official website](https://codalab.lisn.upsaclay.fr/competitions/3627#participate) to get the starting kit, download the public data and set up the environment.

The code was tested with Python 3.7 and Pytorch >= 1.10.0.

## Running code under the competition setting
The details of meta-learner, learner, and predictor can be found in ``model.py``. 
Please follow the [official tutorial](https://github.com/DustinCarrion/cd-metadl) to run the codes.  

Here is an example. First, enter into the ``cd-metadl`` folder, and move the package of cdml22-ltan into the ``baselines`` folder. Then run the following command to train the model and get test results:
```
cd path/to/cd-metadl
python -m cdmetadl.run --submission_dir=path/to/this/folder --output_dir_ingestion=ingestion_output --output_dir_scoring=scoring_output --verbose=False  --test_tasks_per_dataset=10
```
Before running these scripts, please set the dataset directory and  model directory in the arguments or directly change the default directories in the code.
