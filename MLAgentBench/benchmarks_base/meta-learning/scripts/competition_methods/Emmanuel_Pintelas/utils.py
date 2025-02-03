import os
import numpy as np
import random
import torch
import time
from time import gmtime, strftime
import requests
import ot
from torchvision import transforms
# try:
#     import albumentations
# except:
#     os.system("pip install -U albumentations")
# from albumentations import (
#     Compose, HorizontalFlip, CLAHE, HueSaturationValue,
#     RandomBrightness, RandomContrast, RandomGamma,OneOf,
#     ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
#     RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
#     IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, Downscale, IAAPiecewiseAffine, IAASharpen
# )




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

# TRAIN_AUGMENT0 = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.Normalize(127.5,127.5)
# ])

TRAIN_AUGMENT0 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.Normalize(127.5,127.5)
])
TRAIN_AUGMENT1 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomCrop(128, padding=16),
    transforms.Normalize(127.5,127.5)
])
TRAIN_AUGMENT2 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5)
])
TRAIN_AUGMENT3 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomRotation(degrees=(0, 10)),
    transforms.Normalize(127.5,127.5)
])

TRAIN_AUGMENT4 = transforms.Compose([
    transforms.Normalize(-1.0, 2.0/255.0),
    transforms.RandomCrop(128, padding=16),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(127.5,127.5)
])



# TRAIN_AUGMENT = transforms.Compose([

#     transforms.RandomRotation(degrees=(0, 10)),
#     transforms.RandomAdjustSharpness(sharpness_factor=0.5),
#     transforms.RandomAutocontrast(),

# ])


VAL_AUGMENT = transforms.Compose([
            transforms.Normalize(-1.0, 2.0/255.0),
            transforms.Normalize(127.5,127.5)
])



# TRAIN_AUGMENT = albumentations.Compose([
#    albumentations.RandomResizedCrop(50, 50, scale=(0.9, 1), p=1), 
#    albumentations.HorizontalFlip(p=0.5),
#    albumentations.ShiftScaleRotate(p=0.5),
#    albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
#    albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
#    albumentations.CLAHE(clip_limit=(1,4), p=0.5),
#    albumentations.OneOf([
#        albumentations.OpticalDistortion(distort_limit=1.0),
#        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
#        albumentations.ElasticTransform(alpha=3),
#    ], p=0.2),
#    albumentations.OneOf([
#        albumentations.GaussNoise(var_limit=[10, 50]),
#        albumentations.GaussianBlur(),
#        albumentations.MotionBlur(),
#        albumentations.MedianBlur(),
#    ], p=0.2),
#   albumentations.OneOf([
#       JpegCompression(),
#       Downscale(scale_min=0.1, scale_max=0.15),
#   ], p=0.2),
#   IAAPiecewiseAffine(p=0.2),
#   IAASharpen(p=0.2),
#   albumentations.Cutout(max_h_size=int(128 * 0.1), max_w_size=int(128 * 0.1), num_holes=5, p=0.5),
#   albumentations.Normalize(),
# ])

# VAL_AUGMENT = albumentations.Compose([
#     albumentations.Normalize()
# ])






# Normalize(
#          mean=[0.485, 0.456, 0.406],
#          std=[0.229, 0.224, 0.225],
#      )


# # 2
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomGrayscale(p=0.1), # new
#     transforms.Normalize(127.5,127.5)
# ])

# # 3
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=.5, hue=.3), # new
#     transforms.Normalize(127.5,127.5)
# ])
# # 4
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)), # new
#     transforms.Normalize(127.5,127.5)
# ])
# # 5
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomPerspective(distortion_scale=0.6, p=1.0), # new
#     transforms.Normalize(127.5,127.5)
# ])
# # 6
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees=(0, 180)), # new
#     transforms.Normalize(127.5,127.5)
# ])
# # 7
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)), # new
#     transforms.Normalize(127.5,127.5)
# ])
# # 8
# TRAIN_AUGMENT = transforms.Compose([
#     transforms.Normalize(-1.0, 2.0/255.0),
#     transforms.RandomCrop(128, padding=16),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandAugment(), # new
#     transforms.Normalize(127.5,127.5)
# ])


# transforms.RandomGrayscale(p=0.1) 
# ColorJitter(brightness=.5, hue=.3) 
# GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)) 
# RandomPerspective(distortion_scale=0.6, p=1.0)
# RandomRotation(degrees=(0, 180))
# RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
# ElasticTransform(alpha=250.0)
# RandomInvert()
# RandomPosterize(bits=2)
# RandomSolarize(threshold=192.0)
# RandomAdjustSharpness(sharpness_factor=2)
# RandomAutocontrast()
# RandomEqualize()

# RandAugment()


# transforms.Compose([
#         ImageNetPolicy(),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.485, 0.456, 0.406], 
#             [0.229, 0.224, 0.225])
#     ])

def normalize(emb):
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb


def resize_tensor(x,size):
    return transforms.functional.resize(x, [size, size], 
        transforms.functional.InterpolationMode.BILINEAR, antialias=True)


def augment(x,s):
    if s == 0:
        return TRAIN_AUGMENT0(x)
    elif s == 1:
        return TRAIN_AUGMENT1(x)
    elif s == 2:
        return TRAIN_AUGMENT2(x)
    elif s == 3:
        return TRAIN_AUGMENT3(x)
    elif s == 4:
        return TRAIN_AUGMENT4(x)


def augment_valid(x):
    return VAL_AUGMENT(x)

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

    return map_label_propagation(qx, sx)
