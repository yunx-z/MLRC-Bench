<<<<<<< HEAD
=======
#!/usr/bin/env python3
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
# Weather4cast 2023 Starter Kit
#
# This Starter Kit builds on and extends the Weather4cast 2022 Starter Kit,
# the original license for which is included below.
#
# In line with the provisions of this license, all changes and additional
<<<<<<< HEAD
# code are also released unde the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 

=======
# code are also released under the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# 
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
# Weather4cast 2022 Starter Kit
#
# Copyright (C) 2022
# Institute of Advanced Research in Artificial Intelligence (IARAI)
<<<<<<< HEAD
=======
#
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
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
<<<<<<< HEAD

# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran

import argparse
import time
import json
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import torch 

from baseline.unet_lightning_w4c23 import UNet_Lightning as UNetModel
from baseline.utils.data_utils import load_config
from baseline.utils.data_utils import get_cuda_memory_usage
from baseline.utils.data_utils import tensor_to_submission_file
from baseline.utils.w4c_dataloader import RainData

TASK_NAME = "weather4cast-2023"
DEFAULT_METHOD_NAME = "baseline"

def save_evals(task_name, method_name, method_class, base_class, score, phase, runtime):
    """Save evaluation results to a JSON file."""
    eval_data = {
        "task_name": task_name,
        "method_name": method_name,
        "method_class": method_class.__name__ if method_class else None,
        "base_class": base_class.__name__ if base_class else None,
        "score": score,
        "phase": phase,
        "runtime": runtime
    }
    
    os.makedirs("output", exist_ok=True)
    output_file = f"output/{task_name}_{method_name}_{phase}_eval.json"
    with open(output_file, 'w') as f:
        json.dump(eval_data, f, indent=2)
=======
#
# Contributors: Aleksandra Gruca, Pedro Herruzo, David Kreil, Stephen Moran

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy
import datetime
import os
import torch 
import h5py
import numpy as np

# Import from baseline
from baseline.unet_lightning_w4c23 import UNet_Lightning as UNetModel
from baseline.utils.data_utils import load_config, get_cuda_memory_usage, tensor_to_submission_file

# Modified RainData class with lazy loading to avoid OOM issues
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
                region = f.split('.')[0]  # e.g., 'boxi_0015'
                hrit_regions.add(region)
        
        print(f"Found HRIT regions for split {self.split}: {hrit_regions}")
        
        # Get all unique region names from OPERA files
        opera_files = os.listdir(opera_dir) if os.path.exists(opera_dir) else []
        opera_regions = set()
        for f in opera_files:
            if f.endswith(f'.{self.split}.rates.crop.h5'):
                region = f.split('.')[0]
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
            self.regions = [self.matching_regions[0]]
            print(f"Testing with first matching region: {self.regions[0]}")
        
        # Instead of loading all data into memory, we store file paths and determine the sample count.
        self.sat_file = os.path.join(self.hrit_root, str(self.year), 'HRIT', f'{self.regions[0]}.{self.split}.reflbt0.ns.h5')
        if self.split != 'test':
            self.radar_file = os.path.join(self.opera_root, str(self.year), 'OPERA', f'{self.regions[0]}.{self.split}.rates.crop.h5')
        else:
            self.radar_file = None
        
        # Open the satellite file to get shape info and compute sample count
        with h5py.File(self.sat_file, 'r') as f:
            self.num_timesteps = f['REFL-BT'].shape[0]
            self.data_shape = f['REFL-BT'].shape  # (T, channels, height, width)
        if self.split != 'test':
            # For training/validation, each sample consists of 4 input + 32 output timesteps
            self.n_samples = self.num_timesteps - 36
        else:
            # For test, we use non-overlapping windows of 4 timesteps (adjust as needed)
            self.n_samples = (self.num_timesteps - 4) // 4
        
        print(f"Dataset contains {self.n_samples} samples for split {self.split}")
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.split != 'test':
            with h5py.File(self.sat_file, 'r') as sat_f:
                sat_slice = sat_f['REFL-BT'][idx:idx+4]
            with h5py.File(self.radar_file, 'r') as rad_f:
                radar_slice = rad_f['rates.crop'][idx+4:idx+36]
            x = torch.tensor(sat_slice, dtype=torch.float32)  # shape: (4, 11, H, W)
            # Permute to (channels, time, height, width) i.e. (11, 4, H, W)
            x = x.permute(1, 0, 2, 3)
            y = torch.tensor(radar_slice, dtype=torch.float32)
            y = y.permute(1, 0, 2, 3)  # Converts from (32, 1, H, W) to (1, 32, H, W)
            return x, y, {}
        else:
            start_idx = idx * 4
            with h5py.File(self.sat_file, 'r') as sat_f:
                sat_slice = sat_f['REFL-BT'][start_idx:start_idx+4]
            x = torch.tensor(sat_slice, dtype=torch.float32)
            x = x.permute(1, 0, 2, 3)  # Permute similarly for test mode
            _, _, h, w = x.shape
            y = torch.zeros((32, 1, h, w), dtype=torch.float32)
            y = y.permute(1, 0, 2, 3)  # Converts from (32, 1, H, W) to (1, 32, H, W)
            return x, y, {}
        

# The rest of the file remains the same, including DataModule, load_model, get_trainer, etc.
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c

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
        dl = DataLoader(dataset, 
                        batch_size=self.training_params['batch_size'],
                        num_workers=self.training_params['n_workers'],
                        shuffle=shuffle, 
                        pin_memory=pin, prefetch_factor=2,
                        persistent_workers=False)
        return dl
    
    def train_dataloader(self):
        return self.__load_dataloader(self.train_ds, shuffle=True, pin=True)
    
    def val_dataloader(self):
        return self.__load_dataloader(self.val_ds, shuffle=False, pin=True)

    def test_dataloader(self):
        return self.__load_dataloader(self.test_ds, shuffle=False, pin=True)

<<<<<<< HEAD
=======
    def teardown(self, stage=None):
        # Clean up DataLoader workers and free GPU memory
        if hasattr(self, 'train_ds'):
            del self.train_ds
        if hasattr(self, 'val_ds'):
            del self.val_ds
        if hasattr(self, 'test_ds'):
            del self.test_ds
        torch.cuda.empty_cache()

>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c

def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['experiment'], **params['dataset'], **params['train']} 
<<<<<<< HEAD
=======
    
    # Set device and print GPU info
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU Model: {torch.cuda.get_device_name(device)}")
        print(f"Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")
        print(f"  Cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f}MB")
    
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
<<<<<<< HEAD
    return model

def get_trainer(gpus,params):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=3, save_last=True,
                                          filename='{epoch:02d}-{val_loss_epoch:.6f}')
    
    parallel_training = None
    ddpplugin = None   
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
        parallel_training = 'ddp'
##        ddpplugin = DDPPlugin(find_unused_parameters=True)
    print(f"====== process started on the following GPUs: {gpus} ======")
=======
    
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
                valid_gpus = [0]
            # For a single GPU, pass an int
            if len(valid_gpus) == 1:
                devices = 1
                strategy = "single_device"
                print("Using single GPU strategy")
            else:
                devices = valid_gpus
                strategy = "ddp"
                print(f"Using DDP strategy with {len(valid_gpus)} GPUs")
        else:
            devices = 1
            strategy = "single_device"
            print("Using single GPU strategy")

    print(f"====== process started with accelerator: {accelerator}, devices: {devices}, strategy: {strategy} ======")
    
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")
    version = params['experiment']['name']
    version = version + '_' + date_time

<<<<<<< HEAD
    #SET LOGGER 
    if params['experiment']['logging']: 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],name=params['experiment']['sub_folder'], version=version, log_graph=True)
=======
    # Set logger 
    if params['experiment']['logging']: 
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=params['experiment']['experiment_folder'],
                                                name=params['experiment']['sub_folder'], 
                                                version=version, 
                                                log_graph=True)
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    else: 
        tb_logger = False

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]

<<<<<<< HEAD
    trainer = pl.Trainer(devices=gpus, max_epochs=max_epochs,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator="gpu",
                         callbacks=callback_funcs,logger=tb_logger,
                         profiler='simple',precision=params['experiment']['precision'],
                         strategy="ddp"
                        )
=======
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        max_epochs=max_epochs,
        gradient_clip_val=params['model']['gradient_clip_val'],
        gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
        callbacks=callback_funcs,
        logger=tb_logger,
        profiler='simple',
        precision=params['experiment']['precision'],
        strategy=SingleDeviceStrategy(device=0),
        # deterministic=True,
        enable_model_summary=True,
        enable_progress_bar=True,
        auto_select_gpus=True,
        log_every_n_steps=10
    )

    print(f"root_device", trainer.strategy.root_device)
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)   
<<<<<<< HEAD
    tensor_to_submission_file(scores,predict_params)
=======
    tensor_to_submission_file(scores, predict_params)
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)
    
def train(params, gpus, mode, checkpoint_path, model=UNetModel): 
    """ main training/evaluation method
    """
<<<<<<< HEAD
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model, params, checkpoint_path)
    # ------------
    # Add your models here
    # ------------
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params)
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
=======
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
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
        torch.cuda.set_device(gpus[0])
    
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    model = load_model(model, params, checkpoint_path)
    trainer = get_trainer(gpus, params)
    get_cuda_memory_usage(gpus)
    
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    if mode == 'train':
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        trainer.fit(model, data)
    
<<<<<<< HEAD
    
    if mode == "val":
    # ------------
    # VALIDATE
    # ------------
=======
    if mode == "val":
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        do_test(trainer, model, data.val_dataloader()) 

<<<<<<< HEAD

    if mode == 'predict':
    # ------------
    # PREDICT
    # ------------
=======
    if mode == 'predict':
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
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
<<<<<<< HEAD
    config_p = os.path.join('baseline/configurations',options.config_path)
=======
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    if options.mode == 'predict':
        config_file = 'config_baseline_w4c23-pred.yaml'
    else:
        config_file = 'config_baseline_w4c23.yaml'
    config_p = os.path.join(base_path, 'baseline/configurations', config_file)
    print(f"Loading configuration from: {config_p}")
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    params = load_config(config_p)
    
    if options.name != '':
        print(params['experiment']['name'])
        params['experiment']['name'] = options.name
<<<<<<< HEAD
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='./configurations/config_basline.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=1, 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
=======
    
    if options.phase == 'test':
        params['dataset']['data_root'] = os.path.join(params['dataset']['data_root'], 'test')
    
    return params
    
def set_parser():
    parser = argparse.ArgumentParser(description="Weather4cast 2023 Training Script")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=[0], 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-p", "--phase", type=str, required=False, default='dev',
                        help="choose phase: dev (default) / test")
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=False, default='', 
                         help="Set the name of the experiment")
<<<<<<< HEAD
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD_NAME,
                        help="Method name")
    parser.add_argument("--phase", type=str, default="dev",
                        choices=["dev", "test"],
                        help="Phase: dev/test")

=======
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()
<<<<<<< HEAD

    start_time = time.time()
    params = update_params_based_on_args(options)
    train(params, options.gpus, options.mode, options.checkpoint)
    runtime = time.time() - start_time

    # Save evaluation results
    save_evals(
        task_name=TASK_NAME,
        method_name=options.method,
        method_class=UNetModel,
        base_class=pl.LightningModule,
        score=0.0,  # This should be updated with actual model score
        phase=options.phase,
        runtime=runtime,
    )
=======
    params = update_params_based_on_args(options)
    train(params, options.gpus, options.mode, options.checkpoint)
>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c

if __name__ == "__main__":
    main()
    """ examples of usage:
<<<<<<< HEAD

    1) train from scratch on one GPU
    python train.py --gpus 2 --mode train --config_path config_baseline.yaml --name baseline_train

    2) train from scratch on four GPUs
    python train.py --gpus 0 1 2 3 --mode train --config_path config_baseline.yaml --name baseline_train
    
    3) fine tune a model from a checkpoint on one GPU
    python train.py --gpus 1 --mode train  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_tune
    
    4) evaluate a trained model from a checkpoint on two GPUs
    python train.py --gpus 0 1 --mode val  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt" --name baseline_validate

    5) generate predictions (plese note that this mode works only for one GPU)
    python train.py --gpus 1 --mode predict  --config_path config_baseline.yaml  --checkpoint "lightning_logs/PATH-TO-YOUR-MODEL-LOGS/checkpoints/YOUR-CHECKPOINT-FILENAME.ckpt"

    """
=======
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

>>>>>>> 97b09858c83b8c59a7461e7a5f165e9533f92a2c
