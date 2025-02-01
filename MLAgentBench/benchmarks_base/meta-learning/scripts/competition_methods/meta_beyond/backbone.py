import torch
import torch.nn as nn
from torch.nn import Sequential, AdaptiveAvgPool2d, Identity, Module
from typing import Iterable
from torch.nn.modules.flatten import Flatten
import timm
from timm.models.resnet import ResNet
import torch.nn.functional as F


def normalize(x):
    return x / (1e-6 + x.pow(2).sum(dim=-1, keepdim=True).sqrt())


class MLP(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(indim, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, indim // 2), nn.ReLU(),
            nn.Linear(indim // 2, outdim))

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.core(x)


class Wrapper_res(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.fc=nn.Identity()
        self.set = False
    
    def set_get_trainable_parameters(self, parameters=[]):
        params = []
        for i in range(self.num_layers + 1):
            param = self.model.get_parameters(layer=i)
            if i in parameters:
                params.extend(param)
            elif not self.set:
                for p in param:
                    p.requires_grad = False
        self.set = True
        return params

    def set_learnable_layers(self, layers):
        for i in range(self.num_layers + 1):
            self.model.set_layer(i, i in layers)
    
    def set_mode(self, train):
        self.model.set_mode(train)

    def forward(self, x):
        return self.model(x)


class Wrapper_pf(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.set = False
    
    def set_get_trainable_parameters(self, parameters=[]):
        params = []
        for i in range(self.num_layers + 1):
            param = self.model.get_parameters(layer=i)
            if i in parameters:
                params.extend(param)
            elif not self.set:
                for p in param:
                    p.requires_grad = False
        self.set = True
        return params
  
    def set_learnable_layers(self, layers):
        for i in range(self.num_layers + 1):
            self.model.set_layer(i, i in layers)
    
    def set_mode(self, train):
        self.model.set_mode(train)

    def forward(self, x):
        return self.model(x)




def rn_timm_mix(pretrained=True, name='', momentum=0.1):  
    model = timm.create_model(name, pretrained=pretrained)

    return model

def set_parameters(model):
    for name, value in model.named_parameters():
        if ('network.6.' not in name and 'network.4.' not in name):
            value.requires_grad = False

    return model

def set_cls(model):
    trainable_layers = ['model.cls_token']
    for name, value in model.named_parameters():
        if not any([name in layer for layer in trainable_layers]):
            value.requires_grad = False
        else:
            value.requires_grad = True

    return model

def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos',temperature=10):
    n_way = len(query_labels.unique())
    prots = compute_prototypes(support_embeddings, support_labels, n_way).unsqueeze(0)
    embeds = query_embeddings.unsqueeze(1)
    if distance == 'l2':
        logits = -torch.pow(embeds - prots, 2).sum(-1)    # shape [n_query, n_way]
    elif distance == 'cos':
        logits = F.cosine_similarity(embeds, prots, dim=-1, eps=1e-30) * temperature    
    elif distance == 'lin':
        logits = torch.einsum('izd,zjd->ij', embeds, prots)
    elif distance == 'corr':
        logits = F.normalize((embeds * prots).sum(-1), dim=-1, p=2) * 10

    return cross_entropy_loss(logits, query_labels)

def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots

def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict


class EnsembleWrapper(Module):
    def __init__(self, model1,model2):
        super().__init__()
        self.model1=model1
        self.model2=model2
 