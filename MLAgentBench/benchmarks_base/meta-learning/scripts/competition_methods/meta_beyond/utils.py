import os
import numpy as np
import random
import torch
import time
from time import gmtime, strftime
import requests
import ot
from torchvision import transforms
import torch.nn.functional as F

def simple_transform(x, beta):
    x = 1/torch.pow(torch.log(1/x+1),beta)
    return x

def extended_simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg




def mean(x):
    return sum(x) / len(x)


class timer():
    def initialize(self, time_begin='auto', time_limit=60 * 100):
        self.time_limit = time_limit
        self.time_begin = time.time() if time_begin == 'auto' else time_begin
        self.time_list = [self.time_begin]
        self.named_time = {}
        return self

    def anchor(self, name=None, end=None):
        self.time_list.append(time.time())
        if name is not None:
            if name in self.named_time:
                if end:
                    assert self.named_time[name]['time_begin'] is not None
                    self.named_time[name]['time_period'].append(
                        self.time_list[-1] - 
                        self.named_time[name]['time_begin'])
                else:
                    self.named_time[name]['time_begin'] = self.time_list[-1]
            else:
                assert end == False
                self.named_time[name] = {
                    'time_begin': self.time_list[-1],
                    'time_period': []
                }
        return self.time_list[-1] - self.time_list[-2]

    def query_time_by_name(self, name, method=mean, default=50):
        if (name not in self.named_time or 
            self.named_time[name]['time_period'] == []):
            return default
        times = self.named_time[name]['time_period']
        return method(times)

    def time_left(self):
        return self.time_limit - time.time() + self.time_begin
    
    def begin(self, name):
        self.anchor(name, end=False)
    
    def end(self, name):
        self.anchor(name, end=True)
        return self.named_time[name]['time_period'][-1]


DEBUG=0
INFO=1
WARN=2
ERROR=3
LEVEL = DEBUG

_idx2str = ['D', 'I', 'W', 'E']

get_logger = lambda x, filename='log.txt': Logger(x, filename)


class Logger():
    def __init__(self, name='', filename='log.txt') -> None:
        self.name = name
        if self.name != '':
            self.name = '[' + self.name + ']'

        self.debug = self._generate_print_func(DEBUG, filename=filename)
        self.info = self._generate_print_func(INFO, filename=filename)
        self.warn = self._generate_print_func(WARN, filename=filename)
        self.error = self._generate_print_func(ERROR, filename=filename)

    def _generate_print_func(self, level=DEBUG, filename='log.txt'):
        def prin(*args, end='\n'):
            if level >= LEVEL:
                strs = ' '.join([str(a) for a in args])
                str_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                print('[' + _idx2str[level] + '][' + str_time + ']' + 
                    self.name, strs, end=end)
                open(os.path.abspath(os.path.join(os.path.dirname(
                    os.path.abspath(__file__)), '../../' + filename)), 
                    'a').write('[' + _idx2str[level] + '][' + str_time + ']' + 
                    self.name + strs + end
                )
        return prin


def safe_log(url, params):
    try:
        requests.get(url=url, params=params, timeout=1)
    except:
        pass


def map_label_propagation(query, supp, alpha=0.2, n_epochs=20):
    way = len(supp)
    model = GaussianModel(way, supp.device)
    model.initFromLabelledDatas(supp)
    optim = MAP(alpha)
    prob, _ = optim.loop(model, query, n_epochs, None)
    m_estimates = model.estimateFromMask(query, prob)
    return m_estimates

def map_label_propagation_logits(query, supp, alpha=0.2, n_epochs=20):
    way = len(supp)
    model = GaussianModel(way, supp.device)
    model.initFromLabelledDatas(supp)
    
    optim = MAP(alpha)

    prob, _ = optim.loop(model, query, n_epochs, None)
    return prob   


class GaussianModel():
    def __init__(self, n_ways, device):
        self.n_ways = n_ways
        self.device = device

    def to(self, device):
        self.mus = self.mus.to(device)
        
    def initFromLabelledDatas(self, shot_data):
        self.mus = shot_data.mean(dim=1)
        self.mus_origin = shot_data

    def updateFromEstimate(self, estimate, alpha):
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)
    
    def getProbas(self, quer_vec):
        # mus: n_shot * dim
        # quer_vec: n_query * dim
        dist = ot.dist(quer_vec.detach().cpu().numpy(), 
            self.mus.detach().cpu().numpy(), metric="cosine")

        dist=torch.from_numpy(dist).float().to(quer_vec.device)
        
        n_usamples, n_ways = quer_vec.size(0), self.n_ways
        if isinstance(dist, torch.Tensor):
            dist = dist.detach().cpu().numpy()
        p_xj_test = torch.from_numpy(ot.emd(np.ones(n_usamples) / n_usamples, 
            np.ones(n_ways) / n_ways, dist)).float().to(quer_vec.device) * \
            n_usamples
        
        return p_xj_test





    def estimateFromMask(self, quer_vec, mask):
        # mask: queries * ways
        # quer_vec: queries * dim
        return ((mask.permute(1, 0) @ quer_vec) + self.mus_origin.sum(dim=1)) \
            / (mask.sum(dim=0).unsqueeze(1) + self.mus_origin.size(1))


def cross_entropy_loss(logits, targets):
    log_p_y = F.log_softmax(logits, dim=1)
    preds = log_p_y.argmax(1)
    labels = targets.type(torch.long)
    loss = F.nll_loss(log_p_y, labels, reduction='mean')
    acc = torch.eq(preds, labels).float().mean()
    stats_dict = {'loss': loss.item(), 'acc': acc.item()}
    pred_dict = {'preds': preds.cpu().numpy(), 'labels': labels.cpu().numpy()}
    return loss, stats_dict, pred_dict

def compute_prototypes(embeddings, labels, n_way):
    prots = torch.zeros(n_way, embeddings.shape[-1]).type(
        embeddings.dtype).to(embeddings.device)
    for i in range(n_way):
        if torch.__version__.startswith('1.1'):
            prots[i] = embeddings[(labels == i).nonzero(), :].mean(0)
        else:
            prots[i] = embeddings[(labels == i).nonzero(as_tuple=False), :].mean(0)
    return prots



def prototype_loss(support_embeddings, support_labels,
                   query_embeddings, query_labels, distance='cos',temperature=10):
    n_way = len(query_labels.unique())

    prots = support_embeddings.mean(dim=1)
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


def prototype_logits(prototype_embeddings, support_labels,
                   query_embeddings, n_way,temperature=10):

    embeds = query_embeddings.unsqueeze(1)
    logits = F.cosine_similarity(embeds, prototype_embeddings, dim=-1, eps=1e-30) * temperature
    return logits



class MAP:
    def __init__(self, alpha=None):
        self.alpha = alpha
    
    def getAccuracy(self, probas, labels):
        olabels = probas.argmax(dim=1)
        matches = labels.eq(olabels).float()
        acc_test = matches.mean()
        return acc_test
    
    def performEpoch(self, model: GaussianModel, quer_vec, labels):
        m_estimates = model.estimateFromMask(quer_vec, self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        self.probas = model.getProbas(quer_vec)
        if labels is not None:
            acc = self.getAccuracy(self.probas, labels)
            return acc
        return 0.

    def loop(self, model: GaussianModel, quer_vec, n_epochs=20, labels=None):
        
        self.probas = model.getProbas(quer_vec)
        acc_list = []
        if labels is not None:
            acc_list.append(self.getAccuracy(self.probas, labels))
           
        for epoch in range(1, n_epochs+1):
            acc = self.performEpoch(model, quer_vec, labels)
            if labels is not None:
                acc_list.append(acc)
        
        return self.probas, acc_list


TRAIN_AUGMENT = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomCrop(128, padding=16), 
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5)
])


def normalize(emb):
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def resize_tensor(x,size=224):
    return transforms.functional.resize(x, [size, size], 
        transforms.functional.InterpolationMode.BILINEAR, antialias=True)


def augment(x):
    return TRAIN_AUGMENT(x)


def augment_test(x):
    return TRAIN_AUGMENT(x)

def mean(x):
    return sum(x) / len(x)


def whiten(features):
    if len(features.shape) == 3:
        w, s, d = features.shape
        features_2d = features.view(w * s, d)
    else:
        features_2d = features
    features_2d = features_2d - features_2d.mean(dim=0, keepdim=True)
    features_2d = normalize(features_2d)
    if len(features.shape) == 3:
        return features_2d.view(w, s, d)
    return features_2d


def decode_label(sx, qx):
    sx = whiten(sx)
    qx = whiten(qx)

    return map_label_propagation(qx, sx,alpha=0.1,n_epochs=10)


def decode_label_logits(sx, qx):
    sx = whiten(sx)
    qx = whiten(qx)

    return map_label_propagation_logits(qx, sx)


