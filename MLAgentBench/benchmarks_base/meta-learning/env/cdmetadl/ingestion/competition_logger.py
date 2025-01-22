import csv
import numpy as np
from os import makedirs
from typing import Any

from cdmetadl.ingestion.data_generator import Task
from cdmetadl.helpers.scoring_helpers import compute_all_scores


class Logger():
    """ Class to define the logger that can be used by the participants during 
    meta-learning to store detailed logs and get printed messages available in 
    the ingestion output log of the Competition Site. 
    """
    
    def __init__(self, logs_dir: str) -> None:
        """
        Args:
            logs_dir (str): Directory where the logs should be stored.
        """
        self.logs_dir = logs_dir
        self.meta_train_iterations = 0
        self.meta_train_logs_path = f"{self.logs_dir}/meta_train"
        self.meta_validation_iterations = 0
        self.meta_valid_steps = 0
        self.meta_valid_root_path = f"{self.logs_dir}/meta_validation"
        self.print_separator = False
        
    def log(self,
            data: Any,
            predictions: np.ndarray,
            loss: float = None,
            meta_train: bool = True) -> None:
        """ Store the task/batch information, predictions, loss and scores of 
        the current meta-train or meta-validation iteration.

        Args:
            data (Any): Data used to compute predictions, it can be a task or a 
                batch.
            predictions (np.ndarray): Predictions associated to each test 
                example in the specified data. It can be the raw logits matrix 
                (the logits are the unnormalized final scores of your model), 
                a probability matrix, or the predicted labels.
            loss (float, optional): Loss of the current iteration. Defaults to 
                None.
            meta_train (bool, optional): Boolean flag to control if the current 
                iteration belongs to meta-training. Defaults to True.
        """
        # Check the data format
        is_task = False
        if isinstance(data, Task):
            is_task = True
        
        first_log = False
        if meta_train:
            # Create log dirs
            if self.meta_train_iterations == 0:
                first_log = True
                self._create_logs_dirs(self.meta_train_logs_path)
            
            # Print separator after finishing meta-valid step    
            if self.print_separator:
                self.meta_validation_iterations = 0
                self.print_separator = False
                print(f"{'#'*79}\n")
            
            # Prepare paths to files
            self.meta_train_iterations += 1
            ground_truth_path = f"{self.meta_train_logs_path}/ground_truth" 
            predictions_path = f"{self.meta_train_logs_path}/predictions"
            task_file = f"{self.meta_train_logs_path}/tasks.csv"
            performance_file = f"{self.meta_train_logs_path}/performance.csv"
            curr_iter = f"iteration_{self.meta_train_iterations}.out"
            print_text = f"Meta-train iteration {self.meta_train_iterations}:"
        
        else:
            # Create log dirs
            if self.meta_validation_iterations == 0:
                first_log = True
                self.meta_valid_steps += 1
                self.meta_valid_logs_path = f"{self.meta_valid_root_path}/"+ \
                    f"step_{self.meta_valid_steps}"
                makedirs(self.meta_valid_logs_path)
                print(f"\n{'#'*30} Meta-valid step {self.meta_valid_steps} "
                    + f"{'#'*30}")
                self.print_separator = True
                
            self.meta_validation_iterations += 1
            task_file = f"{self.meta_valid_logs_path}/tasks.csv"
            performance_file = f"{self.meta_valid_logs_path}/performance.csv"
            print_text = "Meta-valid iteration " \
                + f"{self.meta_validation_iterations}:"
            
        if is_task:
            # Save task information
            dataset = data.dataset
            N = data.num_ways
            k = data.num_shots
            
            with open(task_file, "a", newline="") as f:
                writer = csv.writer(f)
                if first_log:
                    writer.writerow(["Dataset", "N", "k"])
                writer.writerow([dataset, N, k])
            
            ground_truth = data.query_set[1].cpu().numpy()
            
        else:
            N = None
            ground_truth = data[1].cpu().numpy()
        
        if meta_train:
            # Save ground truth and predicted values
            np.savetxt(f"{ground_truth_path}/{curr_iter}", ground_truth, 
                fmt="%d")
            fmt = "%f" if len(predictions.shape) == 2 else "%d"
            np.savetxt(f"{predictions_path}/{curr_iter}", predictions, fmt=fmt)
        
        # Compute and save performance
        scores = compute_all_scores(ground_truth, predictions, N, not is_task)
        score_names = list(scores.keys())
        score_values = list(scores.values())
        
        if loss is not None:
            score_names.append("Loss")
            score_values.append(loss)
            if is_task:
                print(f"{print_text}"
                    + f"\t{scores['Normalized Accuracy']:.4f} (Normalized "
                    + "Accuracy)"
                    + f"\t{scores['Accuracy']:.4f} (Accuracy)"
                    + f"\t{loss:.4f} (Loss)"
                    + f"\t[{N}-way {k}-shot task from {dataset}]")
            else:
                print(f"{print_text}"
                    + f"\t{scores['Accuracy']:.4f} (Accuracy)"
                    + f"\t{loss:.4f} (Loss)")
        else:    
            if is_task:
                print(f"{print_text}"
                    + f"\t{scores['Normalized Accuracy']:.4f} (Normalized "
                    + "Accuracy)"
                    + f"\t{scores['Accuracy']:.4f} (Accuracy)"
                    + f"\t[{N}-way {k}-shot task from {dataset}]")
            else:
                print(f"{print_text}"
                    + f"\t{scores['Accuracy']:.4f} (Accuracy)")
            
        with open(performance_file, "a", newline="") as f:
            writer = csv.writer(f)
            if first_log:
                writer.writerow(score_names)
            writer.writerow(score_values)
    
    def _create_logs_dirs(self, 
                          dir: str) -> None:
        """ Create all the necessary directories for storing the logs at 
        meta-training time.

        Args:
            dir (str): Directory where the log directories should be created.
        """
        for value_to_log in ["ground_truth", "predictions"]:
            makedirs(f"{dir}/{value_to_log}")
            