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
from pytorch_lightning.plugins import DDPPlugin
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import os
import torch 
import json

from unet_lightning_w4c23 import UNet_Lightning as UNetModel
from utils.data_utils import load_config
from utils.data_utils import get_cuda_memory_usage
from utils.data_utils import tensor_to_submission_file
from utils.w4c_dataloader import RainData

class DataModule(pl.LightningDataModule):
    """ Class to handle training/validation splits in a single object
    """
    def __init__(self, params, training_params, mode):
        super().__init__()
        self.params = params     
        self.training_params = training_params
        if mode in ['train', 'debug']:
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


def load_model(Model, params, checkpoint_path=''):
    """ loads a model from a checkpoint or from scratch if checkpoint_path='' """
    p = {**params['train'], **params['experiment'], **params['dataset']} 
    if checkpoint_path == '':
        print('-> Modelling from scratch!  (no checkpoint loaded)')
        model = Model(params['model'], p)            
    else:
        print(f'-> Loading model checkpoint: {checkpoint_path}')
        model = Model.load_from_checkpoint(checkpoint_path, UNet_params=params['model'], params=p)
    return model

def get_trainer(gpus,params,mode):
    """ get the trainer, modify here its options:
        - save_top_k
     """
    max_epochs=params['train']['max_epochs'];
    print("Trainig for",max_epochs,"epochs");
    checkpoint_callback = ModelCheckpoint(monitor='val_loss_epoch', save_top_k=1, save_last=True,
                                          filename='best')
    
    parallel_training = None
    ddpplugin = None   
    if gpus[0] == -1:
        gpus = None
    elif len(gpus) > 1:
        parallel_training = 'ddp'
##        ddpplugin = DDPPlugin(find_unused_parameters=True)
    print(f"====== process started on the following GPUs: {gpus} ======")
    date_time = datetime.datetime.now().strftime("%m%d-%H:%M")

    #SET LOGGER 
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="output", name="model", version=params['model']['name'], log_graph=True)

    if params['train']['early_stopping']: 
        early_stop_callback = EarlyStopping(monitor="val_loss_epoch",
                                            patience=params['train']['patience'],
                                            mode="min")
        callback_funcs = [checkpoint_callback, early_stop_callback]
    else: 
        callback_funcs = [checkpoint_callback]

    trainer = pl.Trainer(devices=gpus, max_epochs=max_epochs, max_steps=3 if mode == 'debug' else -1,
                         gradient_clip_val=params['model']['gradient_clip_val'],
                         gradient_clip_algorithm=params['model']['gradient_clip_algorithm'],
                         accelerator="gpu",
                         callbacks=callback_funcs,logger=tb_logger,
                         profiler=None,precision=params['experiment']['precision'],
                         strategy="ddp"
                        )

    return trainer

def do_predict(trainer, model, predict_params, test_data):
    scores = trainer.predict(model, dataloaders=test_data)
    scores = torch.concat(scores)   
    tensor_to_submission_file(scores,predict_params)

def do_test(trainer, model, test_data):
    scores = trainer.test(model, dataloaders=test_data)
    return scores[0]
    
def train(params, gpus, mode, checkpoint_path, model=UNetModel): 
    """ main training/evaluation method
    """
    # ------------
    # model & data
    # ------------
    get_cuda_memory_usage(gpus)
    data = DataModule(params['dataset'], params['train'], mode)
    # ------------
    # Add your models here
    # ------------
    
    # ------------
    # trainer
    # ------------
    trainer = get_trainer(gpus, params, mode)
    get_cuda_memory_usage(gpus)
    # ------------
    # train & final validation
    # ------------
    if mode in ['train', 'debug']:
        print("------------------")
        print("--- TRAIN MODE ---")
        print("------------------")
        model = load_model(model, params, checkpoint_path)
        trainer.fit(model, data)
    
    
    if mode in ['train', "val", "debug"]:
    # ------------
    # VALIDATE
    # ------------
        print("---------------------")
        print("--- VALIDATE MODE ---")
        print("---------------------")
        # in train mode, how to load best model here for validation?
        output_dir = os.path.join("output", "model", params['model']['name'])
        best_checkpoint_path = os.path.join(output_dir, "checkpoints", "best.ckpt")
        model = load_model(model, params, best_checkpoint_path)
        scores = do_test(trainer, model, data.val_dataloader()) 
        print(f"Final score (mean CSI): {scores['test_mcsi']}")
        phase = 'dev' if mode in ['train', 'debug'] else 'test'
        result_file = os.path.join(output_dir, f"{phase}_metrics.json")  
        with open(result_file, 'w') as writer:
            json.dump(scores, writer, indent=2)
        print(f"evaluation results saved to {result_file}")


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
    train_config_p = os.path.join('configurations',options.config_path)
    train_params = load_config(train_config_p)
    base_config_p = os.path.join('configurations','config_unmodifiable.yaml')
    base_params = load_config(base_config_p)
    params = dict()
    params['experiment'] = base_params['experiment']
    params['dataset'] = base_params['dataset']
    params['train'] = train_params['train']
    params['model'] = train_params['model']
    
    if options.name != '':
        params['model']['name'] = options.name
    return params
    
def set_parser():
    """ set custom parser """
    
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f", "--config_path", type=str, required=False, default='config_baseline_w4c23.yaml',
                        help="path to config-yaml")
    parser.add_argument("-g", "--gpus", type=int, nargs='+', required=False, default=[0], 
                        help="specify gpu(s): 1 or 1 5 or 0 1 2 (-1 for no gpu)")
    parser.add_argument("-m", "--mode", type=str, required=False, default='train', 
                        help="choose mode: train (default) / val / predict")
    parser.add_argument("-c", "--checkpoint", type=str, required=False, default='', 
                        help="init a model from a checkpoint path. '' as default (random weights)")
    parser.add_argument("-n", "--name", type=str, required=True, default='my_method', 
                         help="Set the name of the method")

    return parser

def main():
    parser = set_parser()
    options = parser.parse_args()

    params = update_params_based_on_args(options)
    if options.mode == 'val':
        params["dataset"]["years"] = ['2020']
    elif options.mode == 'train':
        params["dataset"]["years"] = ['2019']
    train(params, options.gpus, options.mode, options.checkpoint)

if __name__ == "__main__":
    import socket
    import os

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))  # Bind to any free port
            return s.getsockname()[1]

    os.environ["MASTER_PORT"] = str(find_free_port())
    os.environ["MASTER_ADDR"] = "127.0.0.1"

    print(f"Using MASTER_PORT={os.environ['MASTER_PORT']}")
    main()
    """ examples of usage:

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
