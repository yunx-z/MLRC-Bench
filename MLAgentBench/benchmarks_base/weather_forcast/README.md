# Weather4cast 2023 Benchmark

This benchmark is based on the Weather4cast 2023 competition, which focuses on predicting future radar reflectivity from satellite data.

## Task Description

The Weather4cast 2023 competition challenges participants to predict future radar reflectivity from satellite data. The task involves using satellite imagery to predict precipitation patterns, which is crucial for weather forecasting and early warning systems.

## Dataset

The dataset consists of:
- Satellite imagery from various channels
- Ground-truth radar reflectivity data
- Data is organized by regions and time periods

## Evaluation

Models are evaluated based on their ability to predict future radar reflectivity. The primary metric is the Root Mean Square Error (RMSE) between predicted and actual radar reflectivity values.

## Getting Started

1. Explore the data in the `env/data` directory
2. Check the baseline model in `env/methods`
3. Run the main script to train and evaluate models

## Directory Structure

- `env/`: Contains the main code for the benchmark
  - `data/`: Contains the dataset
  - `methods/`: Contains baseline methods and utilities
  - `main.py`: Main script for running the benchmark
  - `evaluation.py`: Code for evaluating model performance
  - `constants.py`: Constants used in the benchmark
- `scripts/`: Contains scripts for setting up and running the benchmark
  - `prepare.py`: Script for preparing the benchmark
  - `research_problem.txt`: Description of the research problem
  - `environment.yml`: Conda environment specification
  - `test_data/`: Test data for the benchmark 