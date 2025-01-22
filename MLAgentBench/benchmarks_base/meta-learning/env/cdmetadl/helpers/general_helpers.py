""" Helper functions to use in the ingestion and scoring programs. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import os
import shutil
import yaml
import json
import re
import pandas as pd
from sklearn.utils import check_random_state
from glob import glob as ls
from typing import List, Tuple


def vprint(message: str, 
           verbose: bool) -> None:
    """ Print a message based on the verbose mode.

    Args:
        message (str): Message to be printed.
        verbose (bool): Verbose mode.
    """
    if verbose: 
        print(message)

 
def exist_dir(dir: str) -> bool:
    """ Check if a directory exists.

    Args:
        dir (str): Directory to be checked.

    Raises:
        NotADirectoryError: Error raised when the directory does not exist.
        
    Returns:
        bool: True if the directory exists, an error is raised otherwise.
    """
    if os.path.isdir(dir):
        return True
    raise NotADirectoryError(f"In exist_dir, directory '{dir}' not found")


def exist_file(file: str) -> bool:
    """ Check if a file exists.

    Args:
        file (str): File to be checked.

    Raises:
        FileNotFoundError: Error raised when the file does not exist.
        
    Returns:
        bool: True if the file exists, an error is raised otherwise.
    """
    if os.path.isfile(file):
        return True
    raise FileNotFoundError(f"In exist_file, file '{file}' not found")
        

def print_list(lst: List[str]) -> None:
    """ Print all the elements inside a list.

    Args:
        lst (List[str]): List to be printed.
    """
    for item in lst:
        print(item)
        
        
def show_dir(dir: str = ".") -> None:
    """ Shows all the files and directories inside the specified directory.

    Args:
        dir (str, optional): Source directory to be listed. Defaults to '.'.
    """
    print(f"{'='*10} Listing directory {dir} {'='*10}")
    print_list(ls(dir))
    print_list(ls(dir + '/*'))
    print_list(ls(dir + '/*/*'))
        
        
def mvdir(source: str, 
          dest: str) -> None:
    """ Move a directory to the specified destination.

    Args:
        source (str): Current directory location.
        dest (str): Directory destination.
        
    Raises:
        OSError: Error raised when the directory cannot be renamed.
    """
    if os.path.exists(source):
        try:
            os.rename(source, dest)
        except:
            raise OSError(f"In mvdir, directory '{source}' could not be moved"
                + f" to '{dest}'")
        
        
def mkdir(dir: str) -> None:
    """ Create a directory. If the directory already exists, deletes it before 
    creating it again.

    Args:
        dir (str): Directory to be created.
        
    Raises:
        OSError: Error raised when the directory cannot not be deleted or 
            created.
    """
    if os.path.exists(dir):
        try:
            shutil.rmtree(dir)
        except:
            raise OSError(f"In mkdir, directory '{dir}' could not be deleted")
    
    try:
        os.makedirs(dir)
    except:
        raise OSError(f"In mkdir, directory '{dir}' could not be created")
        
        
def load_yaml(file: str) -> dict:
    """ Loads the content of a YAML file.

    Args:
        file (str): File in YAML format.

    Raises:
        OSError: Error raised when the file cannot be opened.

    Returns:
        dict: Content of the YAML file.
    """
    try:
        return yaml.safe_load(open(file, "r"))
    except:
        raise OSError(f"In load_yaml, file '{file}' could not be opened or "
            + f"has wrong format")
        

def load_json(file: str) -> dict:
    """ Loads the content of a JSON file.

    Args:
        file (str): File in JSON format.

    Raises:
        OSError: Error raised when the file cannot be opened.

    Returns:
        dict: Content of the JSON file.
    """
    try:
        with open(file, "r") as f:
            return json.loads(f.read())
    except:
        raise OSError(f"In load_json, file '{file}' could not be opened or "
            + f"has wrong format")        
        
        
def check_datasets(input_dir: str, 
                   datasets: List[str],
                   verbose: bool = False) -> dict:
    """ Check the format of all datasets. The expected format can be found in 
    https://github.com/ihsaan-ullah/meta-album/tree/master/DataFormat.
    
    Args:
        input_dir (str): Path to the input data directory.
        datasets (List[str]): Name of the datasets to be checked.
        verbose (bool, optional): Flag to control the verbosity. Defaults to 
            False.
    
    Raises:
        NotADirectoryError: Error raised when a path that is expected to be
            a directory is not.
        FileNotFoundError: Error raised when a path that is expected to be
            a file is not.
        OSError: Error raised when a file cannot be opened.
        Exception: Exception raised when specific columns are not presented
            in the formatted datasets.
            
    Returns:
        dict: Information for all the specified datasets.
    """
    vprint("\t[!] Make sure your datasets follow this format: " 
        + "https://github.com/ihsaan-ullah/meta-album/tree/master/"
        + "DataFormat", verbose)
    
    datasets_info = dict()
    for i, dataset in enumerate(datasets):
        vprint(f"\t\tChecking dataset {i}...", verbose)
        
        # Required paths
        IMAGES_PATH = os.path.join(input_dir, dataset, "images")
        JSON_PATH = os.path.join(input_dir, dataset, "info.json")
        CSV_PATH = os.path.join(input_dir, dataset, "labels.csv")

        # Check paths
        exist_dir(IMAGES_PATH)
        vprint("\t\t\t[+] Image folder", verbose)

        exist_file(JSON_PATH)
        exist_file(CSV_PATH)

        # Read JSON file
        info = load_json(JSON_PATH)
        if "image_column_name" in info:
            IMAGE_COLUMN = info["image_column_name"]
        else:
            vprint("\t\t\t[-] JSON file", verbose)
            raise Exception("In check_datasets, 'image_column_name' not found "
                            + f"in '{JSON_PATH}'")
        if "category_column_name" in info:
            CATEGORY_COLUMN = info["category_column_name"]
        else:
            vprint("\t\t\t[-] JSON file", verbose)
            raise Exception("In check_datasets, 'category_column_name' not "
                            + f"found in '{JSON_PATH}'")
        vprint("\t\t\t[+] JSON file", verbose)
        
        # Read CSV file
        try:        
            data_df = pd.read_csv(CSV_PATH, encoding="utf-8")
        except:
            vprint("\t\t\t[-] CSV file", verbose)
            raise OSError(f"In check_datasets, file '{CSV_PATH}' could not be "
                + f"opened or has wrong format.")

        # Check columns in CSV
        csv_columns = data_df.columns

        # Image
        if IMAGE_COLUMN not in csv_columns:
            vprint("\t\t\t[-] CSV file", verbose)
            raise Exception(f"In check_datasets, column '{IMAGE_COLUMN}' "
                + f"not found in '{CSV_PATH}'")

        # Category
        if CATEGORY_COLUMN not in csv_columns:
            vprint("\t\t\t[-] CSV file", verbose)
            raise Exception(f"In check_datasets, column '{CATEGORY_COLUMN}' "
                + f"not found in '{CSV_PATH}'")
        vprint("\t\t\t[+] CSV file", verbose)
        
        datasets_info[dataset] = (CATEGORY_COLUMN, IMAGE_COLUMN, IMAGES_PATH, 
            CSV_PATH) 
        vprint(f"\t\t[+] Dataset {i}\n", verbose)
    
    return datasets_info


def prepare_datasets_information(input_dir: str,
                                 validation_datasets: int,
                                 seed: int,
                                 verbose: bool=False,
                                 scoring: bool=False) -> Tuple[dict,dict,dict]:
    """ Prepare the required dataset information for the available datasets.

    Args:
        input_dir (str): Path to the input data directory.
        validation_datasets (int): Number of datasets that should be used for 
            meta-validation-
        seed (int): Random seed to be used.
        verbose (bool, optional): Flag to control the verbosity. Defaults to 
            False.
        scoring (bool, optional): Flag to control which script calls this 
            function. If True, only the information of the test datasets is 
            loaded, otherwise, the information of all datasets is loaded. 
            Defaults to False.

    Returns:
        Tuple[dict, dict, dict]: Information for all splits, meta-test, 
            meta-validation and meta-test
    """
    vprint("\tChecking info directory", verbose)
    info_dir = os.path.join(input_dir, "info")
    exist_dir(info_dir)
    
    vprint("\tChecking info splits file", verbose)
    split_file = os.path.join(info_dir, "meta_splits.txt")
    exist_file(split_file)
    
    vprint("\tReading splits file", verbose)
    splits = load_json(split_file)

    test_datasets = splits["meta-test"]
    if scoring:
        train_datasets_info = None
        valid_datasets_info = None
    else:
        train_datasets = splits["meta-train"]
        random_gen = check_random_state(seed)
        if validation_datasets is not None and validation_datasets > 0:
            random_gen.shuffle(train_datasets)
            valid_datasets = train_datasets[:validation_datasets]
            train_datasets = train_datasets[validation_datasets:]
        else:
            valid_datasets = list() 

        vprint("\tChecking train datasets", verbose)
        train_datasets_info = check_datasets(input_dir, train_datasets, verbose)
        
        vprint("\tChecking validation datasets", verbose)
        valid_datasets_info = check_datasets(input_dir, valid_datasets, verbose)
    
    vprint("\tChecking test datasets", verbose)
    test_datasets_info = check_datasets(input_dir, test_datasets, verbose)
    
    return train_datasets_info, valid_datasets_info, test_datasets_info  


def natural_sort(text: str) -> list:
    """ Helper to sort a list of strings with digits in natural order. This
    helper should be used as the key parameter in the sorted() function.

    Args:
        text (str): Text to be processed.

    Returns:
        list: Splitted text into words and digits.
    """
    atoi = lambda x: int(x) if x.isdigit() else x
    return [atoi(c) for c in re.split(r"(\d+)", text)]       
        