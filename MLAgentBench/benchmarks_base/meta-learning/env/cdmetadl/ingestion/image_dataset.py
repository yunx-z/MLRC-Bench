""" Dataset to facilitate the reading of data from the input directory. 
Additionally, it provides the data in Pytorch Tensor format. 

AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
"""
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, List


class ImageDataset(Dataset):

    def __init__(self,
                 datasets_info: dict,
                 img_size: int = 128):
        """
        Args:
            datasets_info (dict): Dictionary with the information of the 
                datasets to be created. For each dataset the required 
                information is (name of the column that has the labels, name of 
                the column that has the file names, path to the images 
                directory, path to the labels.csv file).
            img_size (int, optional): Desired image size. Defaults to 128.
        """
        datasets = list(datasets_info.keys())
        if len(datasets) == 1: 
            self.name = datasets[0]
        else:
            self.name = f"Multiple datasets: {','.join(datasets)}"
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)), transforms.ToTensor()])
            
        self.img_paths = list()
        self.labels = list()
        id_ = 0
        for dataset in datasets:
            (label_col, file_col, img_path, md_path) = datasets_info[dataset]
            metadata = pd.read_csv(md_path)
            self.img_paths.extend([os.path.join(img_path, x) for x in 
                metadata[file_col]])
            
            # Transform string labels into non-negative integer IDs
            label_to_id = dict()
            for label in metadata[label_col]:
                if label not in label_to_id:
                    label_to_id[label] = id_
                    id_ += 1
                self.labels.append(label_to_id[label])
        self.labels = np.array(self.labels)
        
        self.idx_per_label = list()
        self.min_examples_per_class = float("inf")
        for i in range(max(self.labels) + 1):
            idx = np.argwhere(self.labels == i).reshape(-1)
            self.idx_per_label.append(idx)
            if len(idx) < self.min_examples_per_class:
                self.min_examples_per_class = len(idx)

    def __len__(self) -> int:
        """ Retrieves the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Retrieves the element i of the dataset.

        Args:
            i (int): Index of the element to be retrieved.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image and label corresponding to 
                the specified index.
        """
        path, label = self.img_paths[i], self.labels[i]
        image = self.transform(Image.open(path))
        return image, torch.LongTensor([label]).squeeze()
    
    
def create_datasets(datasets_info: dict,
                    img_size: int = 128) -> List[ImageDataset]:
    """ Creates one dataset for each of the keys in datasets_info.

    Args:
        datasets_info (dict): Dictionary with the information of the 
            datasets to be created.
        img_size (int, optional): Desired image size. Defaults to 128.

    Returns:
        List[ImageDataset]: List with all the created datasets.
    """
    img_datasets = list()
    for dataset in datasets_info.keys():
        single_dataset_info = {dataset: datasets_info[dataset]}
        img_dataset = ImageDataset(single_dataset_info, img_size)
        img_datasets.append(img_dataset)
    return img_datasets
