#!/usr/bin/env python3
# Weather4cast 2023 Starter Kit
#
# This Starter Kit builds on and extends the Weather4cast 2022 Starter Kit,
# the original license for which is included below.
#
# In line with the provisions of this license, all changes and additional
# code are also released unde the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 

# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)

# This file is part of the Weather4cast 2022 Starter Kit.
# 
# The Weather4cast 2022 Starter Kit is free software: you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
# The Weather4cast 2022 Starter Kit is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran


import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch 
import h5py
import numpy as np

# Import from baseline
from baseline.unet_lightning_w4c23 import UNet_Lightning as UNetModel
from baseline.utils.data_utils import load_config, get_cuda_memory_usage, tensor_to_submission_file

# Modified RainData class to handle our data paths
class RainData:
    def __init__(self, split, data_root='data', regions=None, year=2019, **kwargs):
        # Use absolute paths based on current file location
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.hrit_root = os.path.join(base_path, 'data', 'w4c23')  # Path for HRIT data
        self.opera_root = os.path.join(base_path, 'data')  # Path for OPERA data
        
        # Convert validation to val for file matching
        self.split = 'val' if split == 'validation' else split
        if self.split == 'training':
            self.split = 'train'  # Convert training to train for file matching
            
        self.year = year
        
        # Find all available regions by scanning directories
        hrit_dir = os.path.join(self.hrit_root, str(self.year), 'HRIT')
        opera_dir = os.path.join(self.opera_root, str(self.year), 'OPERA')
        
        print(f"Scanning HRIT directory: {hrit_dir}")
        print(f"Scanning OPERA directory: {opera_dir}")
        
        # Get all unique region names from HRIT files
        hrit_files = os.listdir(hrit_dir) if os.path.exists(hrit_dir) else []
        hrit_regions = set()
        for f in hrit_files:
            if f.endswith(f'.{self.split}.reflbt0.ns.h5'):
                region = f.split('.')[0]  # Get region name (e.g., 'boxi_0015')
                hrit_regions.add(region)
        
        print(f"Found HRIT regions for split {self.split}: {hrit_regions}")
        
        # Get all unique region names from OPERA files
        opera_files = os.listdir(opera_dir) if os.path.exists(opera_dir) else []
        opera_regions = set()
        for f in opera_files:
            if f.endswith(f'.{self.split}.rates.crop.h5'):
                region = f.split('.')[0]  # Get region name
                opera_regions.add(region)
        
        print(f"Found OPERA regions for split {self.split}: {opera_regions}")
        
        # Find regions that have matching files in both directories
        self.matching_regions = sorted(list(hrit_regions & opera_regions))
        
        if not self.matching_regions:
            raise ValueError(f"No matching files found between HRIT and OPERA for split '{self.split}'. HRIT regions: {hrit_regions}, OPERA regions: {opera_regions}")
        
        print(f"Found {len(self.matching_regions)} regions with matching files: {self.matching_regions}")
        
        # Use only the first matching region for testing
        if regions is not None:
            self.regions = [r for r in regions if r in self.matching_regions]
            if not self.regions:
                raise ValueError(f"None of the specified regions have matching files. Available regions: {self.matching_regions}")
        else:
            # Just use the first matching region
            self.regions = [self.matching_regions[0]]
            print(f"Testing with first matching region: {self.regions[0]}")
        
        self.input_data = None
        self.target_data = None
        
        print(f"Loading data for split: {self.split}")
        print(f"HRIT data from: {self.hrit_root}")
        print(f"OPERA data from: {self.opera_root}")
        
        # Load data for each region
        for region in self.regions:
            # Load satellite data from w4c23/HRIT
            sat_file = os.path.join(self.hrit_root, str(self.year), 'HRIT', f'{region}.{self.split}.reflbt0.ns.h5')
            print(f"Loading satellite data from: {sat_file}")
            
            try:
                with h5py.File(sat_file, 'r') as f:
                    sat_data = f['REFL-BT'][:]
                    print(f"Successfully loaded satellite data with shape: {sat_data.shape}")
            except Exception as e:
                print(f"Error loading satellite data: {e}")
                continue
            
            # Create input sequences
            if self.split != 'test':
                # Load radar data from base OPERA directory
                radar_file = os.path.join(self.opera_root, str(self.year), 'OPERA', f'{region}.{self.split}.rates.crop.h5')
                print(f"Loading radar data from: {radar_file}")
                
                try:
                    with h5py.File(radar_file, 'r') as f:
                        radar_data = f['rates.crop'][:]
                        print(f"Successfully loaded radar data with shape: {radar_data.shape}")
                except Exception as e:
                    print(f"Error loading radar data: {e}")
                    continue
                
                # Create sequences with 4 input timesteps and 32 output timesteps
                for i in range(0, len(sat_data) - 36):  # 4 input + 32 output
                    self.input_data.append(sat_data[i:i+4])  # 4 timesteps input
                    self.target_data.append(radar_data[i+4:i+36])  # 32 timesteps output
            else:
                # For test set, only create input sequences
                for i in range(0, len(sat_data) - 4, 4):
                    self.input_data.append(sat_data[i:i+4])
                    # Add dummy target for consistency
                    self.target_data.append(np.zeros((32, 1, sat_data.shape[2], sat_data.shape[3])))
        
        if not self.input_data:
            raise ValueError("No data was loaded. Please check the data paths and file availability.")
        
        self.input_data = np.array(self.input_data)
        self.target_data = np.array(self.target_data)
        
        print(f"Loaded {len(self.input_data)} sequences for {self.split} split")
        print(f"Input data shape: {self.input_data.shape}")
        print(f"Target data shape: {self.target_data.shape}")
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        with h5py.File(self.sat_file, 'r') as sat_f, h5py.File(self.radar_file, 'r') as rad_f:
            x = torch.tensor(sat_f['REFL-BT'][idx:idx+4], dtype=torch.float32)
            y = torch.tensor(rad_f['rates.crop'][idx+4:idx+36], dtype=torch.float32)
        return x, y, {}


class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train']:
            print("Loading TRAINING/VALIDATION dataset -- as test")
            self.train_ds = RainData('training', **self.params)
            self.val_ds = RainData('validation', **self.params)
            print(f"Training dataset size: {len(self.train_ds)}")
        if mode in ['val']:
            print("Loading VALIDATION dataset -- as test")
            self.val_ds = RainData('validation', **self.params)  
        if mode in ['predict']:    
            print("Loading PREDICTION/TEST dataset -- as test")
            self.test_ds = RainData('test', **self.params)

    def __load_dataloader(self, dataset, shuffle=True, pin=True):
        # Calculate optimal batch size based on GPU memory
        # if torch.cuda.is_available():
        #     gpu = torch.cuda.get_device_properties(0)
        #     total_memory = gpu.total_memory / 1024**3  # Convert to GB
        #     # Use about 40% of available GPU memory
        #     optimal_batch_size = min(
        #         int(total_memory * 0.4 * 1024 / (dataset[0][0].numel() * 4)),  # 4 bytes per float32
        #         self.training_params['batch_size']
        #     )
        #     print(f"Adjusted batch size to {optimal_batch_size} based on GPU memory")
        #     self.training_params['batch_size'] = optimal_batch_size

        # Calculate optimal number of workers
        num_workers = min(
            self.training_params['n_workers'],
            os.cpu_count() or 1,
            self.training_params['batch_size']
        )
        
        dl = DataLoader(
            dataset, 
            batch_size=self.training_params['batch_size'],
            num_workers=num_workers,
            shuffle=shuffle, 
            pin_memory=pin,
            persistent_workers=True,
            prefetch_factor=2,
            generator=torch.Generator(device='cuda') if shuffle else None
        )
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)

    def teardown(self, stage=None):
        # Clean up DataLoader workers and free GPU memory
        if hasattr(self, 'train_ds'):
            del self.train_ds
        if hasattr(self, 'val_ds'):
            del self.val_ds
        if hasattr(self, 'test_ds'):
            del self.test_ds
        torch.cuda.empty_cache()


def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']} 
    
    # Set device and print GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(device)}")
        print(f"Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f}MB")
    
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
    
    # Move model to GPU and print device confirmation
    model = model.to(device)
    print(f"Model moved to device: {device}")
    print(f"Model parameters device: {next(model.parameters()).device}")
    
    # Print model summary
    print("\nModel Summary:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Verify model is on GPU if available
    if torch.cuda.is_available():
        assert next(model.parameters()).is_cuda, "Model is not on GPU!"
        print("Verified model is on GPU")
    
    return model

def get_trainer(gpus, params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs = params['train']['max_epochs']
    print("Training for", max_epochs, "epochs")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    # Configure GPU settings
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU...")
        accelerator = "cpu"
        devices = None
        strategy = "auto"
    else:
        print("CUDA is available. Using GPU...")
        accelerator = "gpu"
        
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        # Print available GPU information
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Validate GPU selection
        if isinstance(gpus, list):
            valid_gpus = [gpu for gpu in gpus if 0 <= gpu < num_gpus]
            if not valid_gpus:
                print("Warning: No valid GPUs specified. Using GPU 0.")
                devices = [0]
            else:
                devices = valid_gpus
        else:
            if 0 <= gpus < num_gpus:
                devices = [gpus]
            else:
                print("Warning: Invalid GPU specified. Using GPU 0.")
                devices = [0]
        
        print(f"Using GPUs: {devices}")
        
        # Set strategy based on number of GPUs
        if len(devices) > 1:
            strategy = "ddp"
            print(f"Using DDP strategy with {len(devices)} GPUs")
        else:
            strategy = "auto"
            print("Using single GPU strategy")

    print(f"====== process started with accelerator: {accelerator}, devices: {devices}, strategy: {strategy} ======")
    
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

    # Set logger 
    if params['experiment']['logging']: 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],
                                                name=params['experiment']['sub_folder'], 
                                                version=version, 
                                                log_graph=True)
    else: 
        tb_logger = False

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        max_epochs=max_epochs,
        gradient_clip_val=params['model']['gradient_clip_val'],
        gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
        callbacks=callback_funcs,
        logger=tb_logger,
        profiler='simple',
        precision=params['experiment']['precision'],
        strategy=strategy,
        # Enable deterministic training for reproducibility
        deterministic=True,
        # Enable automatic optimization
        enable_model_summary=True,
        # Enable progress bar
        enable_progress_bar=True,
        # Log every n steps
        log_every_n_steps=10
    )

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)   
    tensor_to_submission_file(scores,predict_params)

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)
    
def train(params, gpus, mode, checkpoint_path, model=UNetModel): 
    """ main training/evaluation method
    """
    # ------------
    # Initialize CUDA and check GPU availability
    # ------------
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        # Get number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        # Print available GPU information
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Validate GPU selection
        if isinstance(gpus, list):
            valid_gpus = [gpu for gpu in gpus if 0 <= gpu < num_gpus]
            if not valid_gpus:
                print("Warning: No valid GPUs specified. Using GPU 0.")
                gpus = [0]
            else:
                gpus = valid_gpus
        else:
            if 0 <= gpus < num_gpus:
                gpus = [gpus]
            else:
                print("Warning: Invalid GPU specified. Using GPU 0.")
                gpus = [0]
        
        print(f"Using GPUs: {gpus}")
        
        # Set CUDA device
        torch.cuda.set_device(gpus[0])
    
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model, params, checkpoint_path)
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params)
    get_cuda_memory_usage(gpus)
    
    # ------------
    # train & final validation
    # ------------
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)
    
    if mode == "val":
        # ------------
        # VALIDATE
        # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_test(trainer, model, data.val_dataloader()) 

    if mode == 'predict':
        # ------------
        # PREDICT
        # ------------
        print("--------------------")
        print("--- PREDICT MODE ---")
        print("--------------------")
        print("REGIONS!:: ", params["dataset"]["regions"], params["predict"]["region_to_predict"])
        if params["predict"]["region_to_predict"] not in params["dataset"]["regions"]:
            print("EXITING... \"regions\" and \"regions to predict\" must indicate the same region name in your config file.")
        else:
            do_predict(trainer, model, params["predict"], data.test_dataloader())
    
    get_cuda_memory_usage(gpus)

def update_params_based_on_args(options):
    # Use absolute path for configuration files
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    
    # Select config file based on mode
    if options.mode == 'predict':
        config_file = 'config_baseline_w4c23-pred.yaml'
    else:
        config_file = 'config_baseline_w4c23.yaml'
    
    config_p = os.path.join(base_path, 'baseline/configurations', config_file)
    print(f"Loading configuration from: {config_p}")
    params = load_config(config_p)
    
    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
    
    # Update parameters based on phase
    if options.phase == 'test':
        # Use test configuration
        params['dataset']['data_root'] = os.path.join(params['dataset']['data_root'], 'test')
    
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="Weather4cast 2023 Training Script")
    # Config path is now handled internally based on mode
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=[0], 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-p", "--phase", type=str, required=False, default='dev',
                        help="choose phase: dev (default) / test")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='', 
                         help="Set the name of the experiment")

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    train(params, options.gpus, options.mode, options.checkpoint)

if __name__ == "__main__":
    main()
    """ examples of usage:

    1) train from scratch on one GPU (development phase)
    python main.py -m train -p dev --gpus 0 --name baseline_train

    2) train from scratch on four GPUs (development phase)
    python main.py -m train -p dev --gpus 0 1 2 3 --name baseline_train
    
    3) fine tune a model from a checkpoint (development phase)
    python main.py -m train -p dev --gpus 0 --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model (development phase)
    python main.py -m val -p dev --gpus 0 --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (test phase)
    python main.py -m predict -p test --gpus 0 --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"
    """
