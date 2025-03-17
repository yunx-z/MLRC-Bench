#!/usr/bin/env python3
# Weather4cast 2023 Benchmark
# UNet model implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from evaluation import evaluate_prediction

VERBOSE = False

class UNet(pl.LightningModule):
    def __init__(self, in_channels=11, out_channels=32, dropout_rate=0.4, n_blocks=5, start_filts=32):
        super().__init__()
        self.save_hyperparameters()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.start_filts = start_filts
        self.n_blocks = n_blocks
        
        self.down_convs = []
        self.up_convs = []
        
        # Create the encoder pathway
        outs = None
        for i in range(n_blocks):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts * (2**i)
            pooling = True if i < n_blocks-1 else False
            
            down_conv = DownConv(
                ins,
                outs,
                dropout_rate,
                pooling=pooling,
                planar=False,
                activation='relu',
                normalization='batch',
                full_norm=True,
                dim=3,
                conv_mode='same'
            )
            self.down_convs.append(down_conv)
        
        # Create the decoder pathway
        for i in range(n_blocks - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(
                ins,
                outs,
                dropout_rate,
                up_mode='transpose',
                merge_mode='concat',
                planar=False,
                activation='relu',
                normalization='batch',
                dim=3,
                conv_mode='same'
            )
            self.up_convs.append(up_conv)
        
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        
        self.reduce_channels = nn.Conv3d(outs*4, self.out_channels, kernel_size=1, dim=3)
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.apply(self.weight_init)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder pathway
        encoder_outs = []
        for module in self.down_convs:
            x, before_pool = module(x)
            before_pool = self.dropout(before_pool)
            encoder_outs.append(before_pool)
        
        x = self.dropout(x)
        
        # Decoder pathway
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # Final processing
        if VERBOSE: print("pre-reshape", x.shape)
        xs = x.shape
        x = torch.reshape(x, (xs[0], xs[1]*xs[2], 1, xs[3], xs[4]))
        if VERBOSE: print("pre-reduce", x.shape)
        x = self.reduce_channels(x)
        if VERBOSE: print("post-reduce", x.shape)
        xs = x.shape
        x = torch.reshape(x, (xs[0], 1, xs[1], xs[3], xs[4]))
        if VERBOSE: print("post-reshape", x.shape)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y, metadata = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, metadata = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        metrics = evaluate_prediction(y, y_hat)
        
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.log_dict({f'val_{k}': v for k, v in metrics.items()}, on_epoch=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        x, y, metadata = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        metrics = evaluate_prediction(y, y_hat)
        
        self.log('test_loss', test_loss, on_epoch=True)
        self.log_dict({f'test_{k}': v for k, v in metrics.items()}, on_epoch=True)
        return test_loss
    
    def predict_step(self, batch, batch_idx):
        x, y, metadata = batch
        return self(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=2e-2)
        return optimizer

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, pooling=True, planar=False, activation='relu',
                 normalization='batch', full_norm=True, dim=3, conv_mode='same'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.pooling = pooling
        self.normalization = normalization
        self.dim = dim
        
        conv_class = nn.Conv2d if planar else nn.Conv3d
        
        if conv_mode == 'same':
            self.conv1 = conv_class(self.in_channels, self.out_channels, kernel_size=3, padding=1)
            self.conv2 = conv_class(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = conv_class(self.in_channels, self.out_channels, kernel_size=3, padding=0)
            self.conv2 = conv_class(self.out_channels, self.out_channels, kernel_size=3, padding=0)
        
        if self.pooling:
            pool_class = nn.MaxPool2d if planar else nn.MaxPool3d
            self.pool = pool_class(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()
        
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        norm_class = nn.BatchNorm2d if planar else nn.BatchNorm3d
        self.norm0 = norm_class(self.out_channels) if full_norm else nn.Identity()
        self.norm1 = norm_class(self.out_channels)
        
        if VERBOSE:
            if full_norm:
                print("DownConv, full_norm, norm0 =", normalization)
            else:
                print("DownConv, no full_norm")
            print("DownConv, norm1 =", normalization)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.norm0(y)
        y = self.dropout(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.norm1(y)
        y = self.dropout(y)
        y = self.act2(y)
        before_pool = y
        y = self.pool(y)
        
        return y, before_pool

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate, up_mode='transpose', merge_mode='concat',
                 planar=False, activation='relu', normalization='batch', dim=3, conv_mode='same'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.up_mode = up_mode
        self.merge_mode = merge_mode
        self.dim = dim
        
        if up_mode == 'transpose':
            conv_class = nn.ConvTranspose2d if planar else nn.ConvTranspose3d
            self.upconv = conv_class(self.in_channels, self.out_channels, kernel_size=2, stride=2)
        else:
            raise ValueError(f"Unknown up_mode: {up_mode}")
        
        if merge_mode == 'concat':
            conv_in_channels = self.out_channels * 2
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")
        
        conv_class = nn.Conv2d if planar else nn.Conv3d
        if conv_mode == 'same':
            self.conv1 = conv_class(conv_in_channels, self.out_channels, kernel_size=3, padding=1)
            self.conv2 = conv_class(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.conv1 = conv_class(conv_in_channels, self.out_channels, kernel_size=3, padding=0)
            self.conv2 = conv_class(self.out_channels, self.out_channels, kernel_size=3, padding=0)
        
        self.dropout = nn.Dropout3d(dropout_rate)
        
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        
        norm_class = nn.BatchNorm2d if planar else nn.BatchNorm3d
        self.norm0 = norm_class(self.out_channels)
        self.norm1 = norm_class(self.out_channels)
    
    def forward(self, from_up, from_down):
        from_up = self.upconv(from_up)
        
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        
        x = self.conv1(x)
        x = self.norm0(x)
        x = self.dropout(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.act2(x)
        
        return x 