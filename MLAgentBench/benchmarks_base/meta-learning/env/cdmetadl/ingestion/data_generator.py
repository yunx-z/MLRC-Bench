""" Data loader to generate all meta-training, meta-validation, and 
meta-testing tasks for the Cross-Domain MetaDL competition. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import numpy as np
from sklearn.utils import check_random_state
import torch
from typing import Tuple, List, Iterator

from cdmetadl.helpers.general_helpers import vprint
from cdmetadl.ingestion.image_dataset import ImageDataset

# Custom dtypes
TASK_DATA = Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, 
    torch.Tensor, np.ndarray]
SET_DATA = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
RAND_GENERATOR = np.random.mtrand.RandomState


class Task:
    """ Class to define few-shot learning tasks.
    """
    
    def __init__(self, 
                 num_ways: int, 
                 num_shots: int, 
                 support_set: SET_DATA, 
                 query_set: SET_DATA,
                 original_class_idx: np.ndarray, 
                 dataset: str = None) -> None:
        """ 
        Args:
            num_ways (int): Number of ways (classes) in the current task. 
            num_shots (int): Number of shots (images per class) for the support 
                set.
            support_set (SET_DATA): Support set for the current task. The 
                format of the set is (torch.Tensor, torch.Tensor, torch.Tensor)
                where the first tensor corresponds to the images with a shape 
                of [num_ways*num_shots x 3 x 128 x 128]. The second tensor 
                corresponds to the labels with a shape of [num_ways*num_shots].
                The last tensor corresponds to the original labels with a shape 
                of [num_ways*num_shots].
            query_set (SET_DATA): Query set for the current task. The format
                of the set is (torch.Tensor, torch.Tensor, torch.Tensor), where 
                the first tensor corresponds to the images with a shape of 
                [num_ways*query_size x 3 x 128 x 128]. The second tensor 
                corresponds to the labels with a shape of [num_ways*query_size]
                and the last tensor corresponds to the original labels with a 
                shape of [num_ways*num_shots]. The query_size can vary 
                depending on the configuration of the data loader.
            original_class_idx (np.ndarray): Array with the original class 
                indexes used in the current task, its shape is [num_ways, ].
            dataset (str, optional): Name of the dataset used to create the 
                current task. Defaults to None.
        """
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.support_set = support_set
        self.query_set = query_set
        self.original_class_idx = original_class_idx
        self.dataset = dataset
        
        
class CompetitionDataLoader:
    """ Class to define the data loaders, it can be used to initialize the 
    meta-training, meta-validation and meta-testing loaders.
    """
    
    def __init__(self, 
                 datasets: List[ImageDataset], 
                 episodes_config: dict,
                 seed: int,
                 private_info: bool = False,
                 test_generator: bool = False,
                 verbose: bool = False) -> None:
        """
        Args:
            datasets (List[ImageDataset]): List with all the datasets that can
                be used to load the data for the tasks.
            episodes_config (dict): Dictionary with all the required 
                configurations to generate the tasks. The required 
                configurations are: N, min_N, max_N, k, min_k, max_k, and
                query_images_per_class.
            seed (int): Random seed to be used.
            private_info (bool, optional): Flag to control the privacy of 
                the information. If True, the datasets names are kept private, 
                otherwise, the names are shown. Defaults to False.
            test_generator (bool, optional): Flag to control if the generator 
                will be used for meta-testing. If True, the generator will 
                generate the specified number of tasks per dataset, otherwise, 
                the generator only generates the specified number of tasks 
                without guaranteeing uniform number of tasks per dataset. 
                Defaults to False.
            verbose (bool, optional): Flag to control the verbosity of the 
                generator. Defaults to False.

        Raises:
            TypeError: Error raised when the type of an argument is invalid.
            ValueError: Error raised when the value of an argument is invalid.
        """
        class_name = type(self).__name__
        if not isinstance(verbose, bool):
            raise TypeError(f"In {class_name}, only bool is valid argument for"
                + f" verbose. Received: {type(verbose)}")
 
        # Initialize attributes
        self.datasets = datasets
        self.n_way = episodes_config["N"]
        self.min_ways = episodes_config["min_N"]
        self.max_ways = episodes_config["max_N"]
        self.k_shot = episodes_config["k"]
        self.min_shots = episodes_config["min_k"]
        self.max_shots = episodes_config["max_k"]
        self.query_size = episodes_config["query_images_per_class"]
        self.private_info = private_info
        self.test_generator = test_generator
        self.generator = lambda num_tasks: self.generate_tasks(num_tasks, seed)
        
        # Check arguments
        # Check datasets
        vprint("\tInitializing generator...", verbose)
        if not isinstance(datasets, list):
            vprint("\t\t[-] datasets argument", verbose)
            raise TypeError(f"In {class_name}, only list is valid argument for"
                + f" datasets. Received: {type(datasets)}")
        for dataset in datasets:
            if not isinstance(dataset, ImageDataset):
                vprint("\t\t[-] datasets argument", verbose)
                raise TypeError(f"In {class_name}, all datasets must be "
                    + f"ImageDataset. Received: {type(dataset)}")
        vprint("\t\t[+] datasets argument", verbose)
        
        # Check ways config
        if self.n_way is not None:
            if not isinstance(self.n_way, int):
                vprint("\t\t[-] n_way argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for n_way. Received: {type(self.n_way)}")
            if self.n_way < 2:
                vprint("\t\t[-] n_way argument", verbose)
                raise ValueError(f"In {class_name}, n_way cannot be less than "
                    + f"2. Received: {self.n_way}")
        else:
            if not isinstance(self.min_ways, int):
                vprint("\t\t[-] min_ways argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for min_ways. Received: {type(self.min_ways)}")
            if self.min_ways < 1:
                vprint("\t\t[-] min_ways argument", verbose)
                raise ValueError(f"In {class_name}, min_ways cannot be less "
                    + f"than 1. Received: {self.min_ways}")
            if not isinstance(self.max_ways, int):
                vprint("\t\t[-] max_ways argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for max_ways. Received: {type(self.max_ways)}")
            if self.min_ways > self.max_ways:
                vprint("\t\t[-] min_ways argument", verbose)
                raise ValueError(f"In {class_name}, min_ways cannot be greater"
                    + f" than max_ways. Received: {self.min_ways} > " 
                    + f"{self.max_ways}")
        vprint("\t\t[+] n_way argument", verbose)
        vprint("\t\t[+] min_ways argument", verbose)
        vprint("\t\t[+] max_ways argument", verbose)
        
        # Check shots config
        if self.k_shot is not None:
            if not isinstance(self.k_shot, int):
                vprint("\t\t[-] k_shot argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for k_shot. Received: {type(self.k_shot)}")
            if self.k_shot < 1:
                vprint("\t\t[-] k_shot argument", verbose)
                raise ValueError(f"In {class_name}, k_shot cannot be less than"
                    + f" 1. Received: {self.k_shot}")
        else:
            if not isinstance(self.min_shots, int):
                vprint("\t\t[-] min_shots argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for min_shots. Received: {type(self.min_shots)}")
            if self.min_shots < 1:
                vprint("\t\t[-] min_shots argument", verbose)
                raise ValueError(f"In {class_name}, min_shots cannot be less "
                    + f"than 1. Received: {self.min_shots}")
            if not isinstance(self.max_shots, int):
                vprint("\t\t[-] max_shots argument", verbose)
                raise TypeError(f"In {class_name}, only int is valid argument "
                    + f"for max_shots. Received: {type(self.max_shots)}")
            if self.min_shots > self.max_shots:
                vprint("\t\t[-] min_shots argument", verbose)
                raise ValueError(f"In {class_name}, min_shots cannot be "
                    + f"greater than max_shots. Received: {self.min_shots} " 
                    + f"> {self.max_shots}")
            vprint("\t\t[+] k_shot argument", verbose)
            vprint("\t\t[+] min_shots argument", verbose)
            vprint("\t\t[+] max_shots argument", verbose)
        
        # Check query size
        if not isinstance(self.query_size, int):
            vprint("\t\t[-] query_size argument", verbose)
            raise TypeError(f"In {class_name}, only int is valid argument for "
                + f"query_size. Received:{type(self.query_size)}")
        if self.query_size > 20:
            vprint("\t\t[-] query_size argument", verbose)
            raise ValueError(f"In {class_name}, query_size cannot be greater "
                    + f"than 20. Received: {self.query_size}")
        vprint("\t\t[+] query_size argument", verbose)
        
        # Check other args
        if not isinstance(private_info, bool):
            vprint("\t\t[-] private_info argument", verbose)
            raise TypeError(f"In {class_name}, only bool is valid argument for"
                + f" private_info. Received: {type(private_info)}")
        vprint("\t\t[+] private_info argument", verbose)
            
        if not isinstance(test_generator, bool):
            vprint("\t\t[-] test_generator argument", verbose)
            raise TypeError(f"In {class_name}, only bool is valid argument for"
                + f"test_generator. Received: {type(test_generator)}")
        vprint("\t\t[+] test_generator argument", verbose)
        
    def generate_tasks(self, 
                       num_tasks: int, 
                       seed: int) -> Iterator[Task]:
        """ Creates a generator of few-shot learning tasks. 

        Args:
            num_tasks (int): Number of tasks to be generated. If it is not a 
                meta-testing generator this value corresponds to the total 
                number of tasks to be generated, but if it is a meta-testing
                generator, this number corresponds to the number of tasks per 
                dataset to be generated. 
            seed (int): Random seed to be used.

        Yields:
            Iterator[Task]: Generated task.
        """
        random_gen = check_random_state(seed)
        if not self.test_generator:
            for _ in range(num_tasks):
                # Select dataset
                dataset_idx = random_gen.randint(0, len(self.datasets))
                dataset = self.datasets[dataset_idx]
                dataset_name = dataset.name
                if self.private_info:
                    dataset_name = f"Dataset {dataset_idx+1}"
                    
                # Create support and query sets
                (n_way, k_shot, support_x, support_y, support_original_y,
                 query_x, query_y, query_original_y, 
                 unique_labels_idx) = self.create_support_and_query_sets(
                     random_gen, dataset)
                
                # Return the task
                task = Task(n_way, k_shot, (support_x, support_y, 
                    support_original_y), (query_x, query_y, query_original_y), 
                    unique_labels_idx, dataset_name)
                yield task        
        else:
            for dataset_idx in range(len(self.datasets)):
                # Dataset information
                dataset = self.datasets[dataset_idx]
                dataset_name = dataset.name
                if self.private_info:
                    dataset_name = f"Dataset {dataset_idx+1}"
                for _ in range(num_tasks):
                    # Create support and query sets
                    (n_way, k_shot, support_x, support_y, support_original_y,
                     query_x, query_y, query_original_y, 
                     unique_labels_idx) = self.create_support_and_query_sets(
                         random_gen, dataset)

                    # Return the task
                    task = Task(n_way, k_shot, (support_x, support_y, 
                        support_original_y), (query_x, query_y, 
                        query_original_y), unique_labels_idx, dataset_name)
                    yield task    
       
    def create_support_and_query_sets(self,
                                      random_gen: RAND_GENERATOR,
                                      dataset: ImageDataset) -> TASK_DATA:
        """ Creates the support and query set for a task using the specified 
        dataset.

        Args:
            random_gen (RAND_GENERATOR): Random generator to be used.
            dataset (ImageDataset): Dataset from wich the task should be carved 
                out.

        Returns:
            TASK_DATA: All the information of the created task.
        """
        # Extract dataset info
        idx_per_label = dataset.idx_per_label
        num_classes = len(idx_per_label)
        min_examples_per_class = dataset.min_examples_per_class
        
        # Select task configuration
        n_way, k_shot = self.prepare_task_config(random_gen, num_classes, 
            min_examples_per_class - self.query_size)
        support_size = n_way * k_shot
        total_examples_per_class = k_shot + self.query_size
        
        # Select examples for the task
        idx_per_class = list()
        classes = random_gen.permutation(num_classes)[:n_way]
        for c in classes:
            available_examples = idx_per_label[c]
            selected_examples = random_gen.choice(available_examples, 
                total_examples_per_class, replace=False)
            idx_per_class.append(selected_examples)
        idx_per_class = np.stack(idx_per_class).T.reshape(-1)
        
        # Load the examples
        data = list()
        original_labels_idx = list()
        for idx in idx_per_class:
            img, label = dataset[idx]
            data.append(img)
            original_labels_idx.append(label)
        data = torch.stack(data)
        labels = torch.arange(n_way).repeat((
            n_way * total_examples_per_class)//n_way).long()
        original_labels_idx = torch.stack(original_labels_idx).long()
        unique_labels_idx = original_labels_idx.numpy()[:n_way]
        
        # Split support and query sets
        (support_x, support_y, support_original_y, 
         query_x, query_y, query_original_y) = (data[:support_size], 
            labels[:support_size], original_labels_idx[:support_size], 
            data[support_size:], labels[support_size:], 
            original_labels_idx[support_size:])  
        
        shuffle_support = random_gen.permutation(support_size) 
        support_x = support_x[shuffle_support]
        support_y = support_y[shuffle_support]
        support_original_y = support_original_y[shuffle_support]
        shuffle_query = random_gen.permutation(len(query_x)) 
        query_x = query_x[shuffle_query]
        query_y = query_y[shuffle_query]
        query_original_y = query_original_y[shuffle_query]
        
        return n_way, k_shot, support_x, support_y, support_original_y, \
            query_x, query_y, query_original_y, unique_labels_idx
             
    def prepare_task_config(self, 
                            random_gen: RAND_GENERATOR,
                            max_ways: int, 
                            max_shots: int) -> Tuple[int, int]:
        """ Prepares the number of ways and shots for a few-shot learning task.

        Args:
            random_gen (RAND_GENERATOR): Random generator to be used.
            max_ways (int): Number of classes available in the current dataset.
            max_shots (int): Minimum number of images per class available in 
                the current dataset.

        Returns:
            Tuple[int, int]: Number of ways and shots for a few-shot learning 
                task.
        """
        n_way = self.n_way 
        if n_way is None:
            if self.max_ways < max_ways:
                max_ways = self.max_ways
            
            if self.min_ways > max_ways:
                ways_variability = self.max_ways - self.min_ways
                min_ways = max_ways - ways_variability
                if min_ways < 2:
                    min_ways = 2
            else:
                min_ways = self.min_ways
            
            n_way = random_gen.randint(min_ways, max_ways + 1)
        else:
            if n_way > max_ways:
                n_way = max_ways
        
        k_shot = self.k_shot
        if k_shot is None:
            if self.max_shots < max_shots:
                max_shots = self.max_shots
            
            if self.min_shots > max_shots:
                shots_variability = self.max_shots - self.min_shots
                min_shots = max_shots - shots_variability
                if min_shots < 1:
                    min_shots = 1
            else:
                min_shots = self.min_shots
                
            k_shot = random_gen.randint(min_shots, max_shots+1)
        else:
            if k_shot > max_shots:
                k_shot = max_shots
        
        return n_way, k_shot
                