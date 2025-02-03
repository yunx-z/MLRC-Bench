'''

This code allows you to introduce multiple light-weight task adaptation modules and quickly learn 
task-adaptive weights from scratch for each meta-test task.
'''

from operator import mod
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
from loss import prototype_loss
 
import copy
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
 
 
class conv_lta(nn.Module):
    def __init__(self, orig_conv):
        super(conv_lta, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific parameters
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1).cuda())
        self.alpha.requires_grad = True
 

    def forward(self, x):
        y = self.conv(x)   
        z=F.conv2d(x, self.alpha, stride=self.conv.stride)
        y=y+z
        return y

class Adapter(nn.Module):
    """ 
    define a task-specific linear adapter and attach this layer to the final head layer of the model.
    """
    def __init__(self, feat_dim):
        super(Adapter, self).__init__()
        
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))
        self.weight.requires_grad = True
     

    def forward(self, x): 
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
 
        return F.conv2d(x,  self.weight.to(x.device)).flatten(1)
       

class poolformer_lta(nn.Module):
    def __init__(self,orig_model):
        """
        append a task-specific linear adapter after the final head layer of the PoolFormer backbone.
        """
        super(poolformer_lta,self).__init__()
        for k,v in orig_model.named_parameters():
            v.requires_grad=False
            
        self.backbone = orig_model
        self.adapter=Adapter(1000)
        setattr(self, 'adapter', self.adapter)


    def forward(self,x):
        return self.backbone.forward(x)

    def reset(self):
      # initialize the task-specific adapter.
        v = self.adapter.weight
        self.adapter.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)




class resnet_lta(nn.Module):
    """  Insert a small number of task-specific weights into the convolutional layers
     and append a task-specific linear adapter after the final layer of the ResNet50 backbone."""
    def __init__(self, orig_resnet):
        super(resnet_lta, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

     
        # insert task-specific parameters       
        for block in orig_resnet.layer2:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_lta(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer3:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_lta(m)
                    setattr(block, name, new_conv)

        for block in orig_resnet.layer4:
            for name, m in block.named_children():
                if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                    new_conv = conv_lta(m)
                    setattr(block, name, new_conv)

        self.backbone = orig_resnet
       
        # attach the task-specific adapter
        feat_dim = orig_resnet.layer4[-1].bn3.num_features 
        adapter = Adapter(feat_dim)
        setattr(self, 'adapter', adapter)
     
    def forward(self, x):
        return self.backbone.forward(x)

    def embed(self, x):
        return self.backbone.embed(x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
    # initialize task adaptation modules
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001
       
        v = self.adapter.weight
        self.adapter.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)

 

def lta(context_images, context_labels,model_pf,model_res, max_iter=13,lr_adapter_pf=1.1, lr_alpha=0.4, lr_adapter_res=3.3, distance='cos'):
    
    """
    Freeze the backbone models and only update parameters of  the task adaptation modelues.
    """
    model_pf.eval()
    model_res.eval()
    
    # setting the optimizer for task adaptation modules
    adapter_params1 = [v for k, v in model_pf.named_parameters() if 'adapter' in k]
    alpha_parms2=[v for k,v in model_res.named_parameters() if 'alpha' in k]
    adapter_params2 = [v for k, v in model_res.named_parameters() if 'adapter' in k]
    params = []
    params.append({'params': adapter_params1, 'lr': lr_adapter_pf})
    params.append({'params':alpha_parms2,'lr':lr_alpha})
    params.append({'params':adapter_params2,'lr':lr_adapter_res})
    optimizer = torch.optim.Adadelta(params, lr=1) 

    context_labels=context_labels.cuda()
    context_images=context_images.cuda()
    scaler=GradScaler()


    # learn the task adaptation modules
    # Automatic Mixed Precision to help reduce memory cost
    for i in range(max_iter):     
        optimizer.zero_grad()
        model_pf.zero_grad()
        model_res.zero_grad()

        with autocast():
            context_features1 = model_pf(context_images)      
            aligned_features1 = model_pf.adapter(context_features1) 
            context_features2=model_res(context_images)
            aligned_features2=model_res.adapter(context_features2)

            loss1, _, _ = prototype_loss(aligned_features1, context_labels,
                                       aligned_features1, context_labels, distance=distance)
            loss2, _, _ = prototype_loss(aligned_features2, context_labels,
                                       aligned_features2, context_labels, distance=distance)

        loss=loss1+loss2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()      
    return