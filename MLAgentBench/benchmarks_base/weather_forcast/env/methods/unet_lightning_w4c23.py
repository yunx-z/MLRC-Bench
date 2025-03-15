#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# UNet Lightning model for the benchmark

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

class UNet_Lightning(pl.LightningModule):
    def __init__(self, UNet_params, params):
        super().__init__()
        self.save_hyperparameters()
        
        self.UNet_params = UNet_params
        self.params = params
        
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.seq_len = self.params['in_seq_len']
        self.pred_len = self.params['out_seq_len']
        self.criterion = nn.MSELoss()
        
        # Initialize validation metrics
        self.validation_step_outputs = []
        
        # UNet architecture
        self.encoder = Encoder(
            in_channels=self.UNet_params['in_channels'],
            out_channels=self.UNet_params['init_features'],
            depth=self.UNet_params['depth']
        )
        
        self.decoder = Decoder(
            in_channels=self.UNet_params['init_features'] * 2**(self.UNet_params['depth']-1),
            out_channels=self.UNet_params['out_channels'],
            depth=self.UNet_params['depth']
        )
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels, height, width]
        batch_size, seq_len, channels, height, width = x.size()
        
        # Reshape for encoder
        x = x.view(batch_size, seq_len * channels, height, width)
        
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        output = self.decoder(features)
        
        # Reshape output to sequence
        output = output.view(batch_size, self.pred_len, 1, height, width)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.validation_step_outputs.append(val_loss)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss
    
    def on_validation_epoch_start(self):
        # Clear the validation step outputs at the start of each validation epoch
        self.validation_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Calculate epoch-level validation loss
        epoch_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val_loss_epoch', epoch_loss, on_epoch=True, prog_bar=True, logger=True)
        # Clear the outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return test_loss
    
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self(x)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.params['max_epochs'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        
        # Initial convolution
        self.down_path.append(
            DoubleConv(in_channels, out_channels)
        )
        
        # Downsampling path
        for i in range(depth-1):
            self.down_path.append(
                Down(out_channels * 2**i, out_channels * 2**(i+1))
            )
    
    def forward(self, x):
        features = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            features.append(x)
            if i < self.depth-1:
                x = F.max_pool2d(x, kernel_size=2)
        
        return features


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        self.depth = depth
        self.up_path = nn.ModuleList()
        
        # Upsampling path
        for i in range(depth-1):
            self.up_path.append(
                Up(in_channels // 2**i, in_channels // 2**(i+1))
            )
        
        # Final convolution
        self.final_conv = nn.Conv2d(in_channels // 2**(depth-1), out_channels, kernel_size=1)
    
    def forward(self, features):
        x = features[-1]
        
        for i, up in enumerate(self.up_path):
            x = up(x, features[-(i+2)])
        
        return self.final_conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x) 