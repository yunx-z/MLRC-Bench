import os
import time
import datetime
from pprint import pprint

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from libs.datasets import make_dataset, make_data_loader
from libs.utils import (train_one_epoch,
                     save_checkpoint, make_optimizer, make_scheduler,
                     fix_random_seed, ModelEma)

def train_model(method):
    """Training entry point called from main.py
    
    Args:
        method: MyMethod instance
        
    Returns:
        Path to saved checkpoint
    """
    # Get model and training config
    model, cfg = method.run("train")
    pprint(cfg)

    # Move model to GPU
    model = model.cuda()
    
    # Setup output folders - ensure root ckpt folder exists
    root_ckpt = "ckpt"
    if not os.path.exists(root_ckpt):
        os.makedirs(root_ckpt)
        
    # Create method-specific folder
    method_name = method.get_name()
    ckpt_folder = os.path.join(root_ckpt, method_name)
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
        
    # Setup tensorboard
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))
    
    # Fix random seeds
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    
    # Scale learning rate / workers for multi-GPU
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])
    
    # Create train dataset and loader
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    
    # Update config based on dataset attributes
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']
    
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    
    # Create optimizer and scheduler
    optimizer = make_optimizer(model, cfg['opt'])
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    
    # Enable model EMA
    print("Using model EMA...")
    model_ema = ModelEma(model)

    # Handle resuming from checkpoint
    start_epoch = 0
    if getattr(cfg, 'resume', None):  # Only resume if specified in config
        resume_path = cfg['resume']
        if os.path.isfile(resume_path):
            print(f"=> Loading checkpoint '{resume_path}'")
            checkpoint = torch.load(
                resume_path,
                map_location=lambda storage, loc: storage.cuda(cfg['devices'][0])
            )
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> Loaded checkpoint '{resume_path}' (epoch {start_epoch})")
            del checkpoint
        else:
            print(f"=> No checkpoint found at '{resume_path}'")
            return None  # Exit if resume path specified but file not found
    
    # Save initial config for reference
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        
    # Calculate max epochs - handle missing warmup_epochs
    warmup_epochs = cfg['opt'].get('warmup_epochs', 0)  # default to 0 if not specified
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + warmup_epochs
    )
    print(f"\nWill train for total {max_epochs} epochs")
    print(f"\nStart training model {cfg['model_name']} from epoch {start_epoch}...")
    
    # Training loop
    for epoch in range(start_epoch, max_epochs):
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=cfg['train_cfg'].get('print_freq', 10)
        )
        
        # Save checkpoint at specified frequency
        ckpt_freq = cfg.get('ckpt_freq', 5)  # Default to every 5 epochs
        if ((epoch + 1) == max_epochs) or (ckpt_freq > 0 and (epoch + 1) % ckpt_freq == 0):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'state_dict_ema': model_ema.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            
            # Save epoch checkpoint
            save_checkpoint(
                save_states, 
                False,
                file_folder=ckpt_folder,
                file_name=f'epoch_{epoch+1:03d}.pth.tar'
            )
            
            # Also save as model_best.pth.tar for validation/testing
            if (epoch + 1) == max_epochs:
                checkpoint_path = os.path.join(ckpt_folder, 'model_best.pth.tar')
                save_checkpoint(
                    save_states,
                    False,
                    file_folder=ckpt_folder,
                    file_name='model_best.pth.tar'
                )
    
    # Cleanup
    tb_writer.close()
    print(f"Training completed. Last checkpoint saved at epoch {max_epochs}")
    
    return checkpoint_path