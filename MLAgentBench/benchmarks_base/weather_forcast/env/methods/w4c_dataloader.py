#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# Dataloader for the benchmark

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_utils import create_sequences, normalize_data

class RainData(Dataset):
    """Dataset class for loading Weather4cast data."""
    
    def __init__(self, split, data_root, regions, year, in_seq_len, out_seq_len, 
                 sat_channels=None, normalize=True, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            split (str): Data split ('train', 'validation', or 'test')
            data_root (str): Root directory of the data
            regions (list): List of regions to use
            year (int): Year of the data
            in_seq_len (int): Length of input sequence
            out_seq_len (int): Length of output sequence
            sat_channels (list): List of satellite channels to use
            normalize (bool): Whether to normalize the data
        """
        super().__init__()
        self.split = 'val' if split == 'validation' else split
        self.data_root = data_root
        self.regions = regions
        self.year = year
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.sat_channels = sat_channels
        self.normalize = normalize
        
        # Load data
        self.input_data = []
        self.target_data = []
        
        for region in self.regions:
            # Load satellite data
            sat_file = os.path.join(self.data_root, str(self.year), 'HRIT', 
                                   f'{region}.{self.split}.reflbt0.ns.h5')
            with h5py.File(sat_file, 'r') as f:
                sat_data = f['REFL-BT'][:]
            
            # Select channels if specified
            if self.sat_channels is not None:
                channel_indices = [i for i, ch in enumerate(
                    ['IR_016', 'IR_039', 'IR_087', 'IR_097', 'IR_108', 'IR_120',
                     'IR_134', 'VIS006', 'VIS008', 'WV_062', 'WV_073']
                ) if ch in self.sat_channels]
                sat_data = sat_data[:, channel_indices]
            
            # Normalize satellite data
            if self.normalize:
                sat_data = normalize_data(sat_data)
            
            # Create input sequences
            if self.split != 'test':
                # Load radar data
                radar_file = os.path.join(self.data_root, str(self.year), 'OPERA', 
                                         f'{region}.{self.split}.rates.crop.h5')
                with h5py.File(radar_file, 'r') as f:
                    radar_data = f['rates.crop'][:]
                
                # Create sequences
                for i in range(0, len(sat_data) - self.in_seq_len - self.out_seq_len + 1):
                    self.input_data.append(sat_data[i:i+self.in_seq_len])
                    self.target_data.append(radar_data[i+self.in_seq_len:i+self.in_seq_len+self.out_seq_len])
            else:
                # For test set, only create input sequences
                for i in range(0, len(sat_data) - self.in_seq_len + 1, self.in_seq_len):
                    self.input_data.append(sat_data[i:i+self.in_seq_len])
                    # Add dummy target for consistency
                    self.target_data.append(np.zeros((self.out_seq_len, 1, sat_data.shape[2], sat_data.shape[3])))
        
        # Convert to numpy arrays
        self.input_data = np.array(self.input_data)
        self.target_data = np.array(self.target_data)
        
        print(f"Loaded {len(self.input_data)} sequences for {split} split")
        print(f"Input shape: {self.input_data.shape}")
        print(f"Target shape: {self.target_data.shape}")
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        # Convert to torch tensors
        input_tensor = torch.from_numpy(self.input_data[idx]).float()
        target_tensor = torch.from_numpy(self.target_data[idx]).float()
        
        # Reshape if needed
        if len(input_tensor.shape) == 3:
            # Add channel dimension
            input_tensor = input_tensor.unsqueeze(1)
        
        if len(target_tensor.shape) == 3:
            # Add channel dimension
            target_tensor = target_tensor.unsqueeze(1)
        
        return input_tensor, target_tensor 