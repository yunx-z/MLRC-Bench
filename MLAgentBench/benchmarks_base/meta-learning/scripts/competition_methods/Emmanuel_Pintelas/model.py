"""Our approach has the following main key characteristics:
         - pre-trained Seresnet152d base model with size input 224x224
         - Meta-Train dataset is utilized
         - We apply "Circular Augmentations" during Meta-Train.
         - We apply a new training scheduler pipeline during Meta-Train.
         - In Meta-Test phase we utilize an ensemble of Linear-based and Distance-based ML models. 
           In this step, we use the Support-Set to extract features via the Seresnet152d baseline 
           and then feed them into the proposed ensemble ML classifier.

Our research contributions:
         - We introduce "Circular Augmentations" which is an augmentation pipeline scheduler 
           in order to improve the training of any CNN-based model
         - We introduce a new training scheduler pipeline which is an optimization validation scheduler 
           in order to improve the training of any CNN-based model.
         - We propose an ensemble of Linear-based and Distance-based ML models 
           which drastically improves the final classification performance specifically for the Any-Way-Any-Shot Learning tasks.
"""



import pickle
import time
import random

TIME_LIMIT = 12000 # time limit of the whole process in seconds 4500
TIME_TRAIN = TIME_LIMIT - 30*60 # set aside 30min for test
t1 = time.time()

import os
import torch

try:
    import numpy as np
except:
    os.system("pip install numpy")

try:
    import cython
except:
    os.system("pip install cython")

try:
    import ot
except:
    os.system("pip install POT")

try:
    import tqdm
except:
    os.system("pip install tqdm")

try:
    import timm
except:
    os.system("pip install timm")

from utils import get_logger, timer, resize_tensor, augment, augment_valid, decode_label, mean
from api import MetaLearner, Learner, Predictor
from backbone import MLP, rn_timm_mix, Wrapper
from torch import optim
import torch.nn.functional as F
from typing import Iterable, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier # PassiveAggressiveClassifier(max_iter=1000, random_state=0) # 0.508
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import tensorflow as tf
from skimage.color import rgb2gray
from skimage import exposure
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


# try:
#     import albumentations
# except:
#     os.system("pip install -U albumentations")


# import albumentations
# from albumentations import (
#     Compose, HorizontalFlip, CLAHE, HueSaturationValue,
#     RandomBrightness, RandomContrast, RandomGamma,OneOf,
#     ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
#     RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
#     IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, Downscale, IAAPiecewiseAffine, IAASharpen
# )






#tf.random.set_seed(2)
# --------------- MANDATORY ---------------
SEED = 98
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)    
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
# -----------------------------------------

LOGGER = get_logger('GLOBAL')
DEVICE = torch.device('cuda') # cuda

class MyMetaLearner(MetaLearner):

    def __init__(self, 
                 train_classes: int, 
                 total_classes: int,
                 logger: Any) -> None:

        super().__init__(train_classes, total_classes, logger)
        
        self.timer = timer()
        self.timer.initialize(time.time(), TIME_TRAIN - time.time() + t1)
        self.timer.begin('load pretrained model')
        self.model = Wrapper(rn_timm_mix(True, 'seresnet152d', 
            0.1)).to(DEVICE)
        
        times = self.timer.end('load pretrained model')
        LOGGER.info('current model', self.model)
        LOGGER.info('load time', times, 's')
        self.dim = 2048

        # only optimize the last 2 layers
        backbone_parameters = []
        backbone_parameters.extend(self.model.set_get_trainable_parameters([3, 
            4]))
        # set learnable layers
        self.model.set_learnable_layers([3, 4])
        self.cls = MLP(self.dim, train_classes).to(DEVICE)
        self.opt = optim.Adam(
            [
                {"params": backbone_parameters},
                {"params": self.cls.parameters(), "lr": 1e-3}
            ], lr=1e-4
        )

    def meta_fit(self, 
                 meta_train_generator: Iterable[Any],
                 meta_valid_generator: Iterable[Any]) -> Learner:
        # fix the valid dataset for fair comparison
        valid_task, n_shots, n_ways, supp_end = [], [], [], []
        L = 50  # 
        for task in meta_valid_generator(L):
            # fixed 5-way 5-shot 5-query settings
            supp_x, supp_y = task.support_set[0], task.support_set[1]
            quer_x, quer_y = task.query_set[0], task.query_set[1]
            n_shots.append(task.num_shots)
            n_ways.append(task.num_ways)

            ##supp_x, quer_x = augment_valid(supp_x), augment_valid(quer_x)

            supp_x = supp_x[supp_y.sort()[1]]
            supp_end .append(supp_x.size(0))
            valid_task.append([torch.cat([resize_tensor(supp_x, 224), resize_tensor(quer_x, 224)]), quer_y])

        # loop until time runs out
        total_epoch = 0

        # eval ahead
        start_time = time.time()
        with torch.no_grad():
            self.model.set_mode(False)
            acc_valid = 0
            for i in range (L):

                x_224, quer_y = valid_task[i]
                n_sh, n_w = n_shots[i], n_ways[i]

                x = x_224.to(DEVICE)
                x = self.model(x)
                supp_x, quer_x = x[:supp_end[i]], x[supp_end[i]:]

                supp_x = supp_x.view(n_w, n_sh, supp_x.size(-1))
                logit = decode_label(supp_x, quer_x).cpu().numpy()
                acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
            acc_valid /= len(valid_task)
            LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch,
                acc_valid))

        best_valid = acc_valid
        best_param = pickle.dumps(self.model.state_dict())
        best_param_2nd = best_param
        print('init_best_valid: ', best_valid)

        self.cls.train()
        cnt, cnt2, s = 0, 0, 0
        while self.timer.time_left() > 60 * 5:
            # train loop
            self.model.set_mode(True)
            for _ in range(5):
                total_epoch += 1
                self.opt.zero_grad()
                err = 0
                acc = 0
                for i, batch in enumerate(meta_train_generator(10)):
                    self.timer.begin('train data loading')
                    X_train, y_train = batch


                    # Cyrcling Augmentation
                    if s == 0:
                        X_train = augment(X_train,0) #2 augms
                        s += 1
                    elif s == 1:
                        X_train = augment(X_train,1) #2 augms + 1 new
                        s += 1
                    elif s == 2:
                        X_train = augment(X_train,2) #2 augms + 1 new
                        s += 1
                    elif s == 3:
                        X_train = augment(X_train,3) #2 augms + 1 new
                        s += 1
                    elif s == 4:
                        X_train = augment(X_train,4) #2 augms + 1 new
                        s = 0


                    X_train = resize_tensor(X_train, 224)
                    X_train = X_train.to(DEVICE)
                    y_train = y_train.view(-1).to(DEVICE)
                    self.timer.end('train data loading')

                    self.timer.begin('train forward')
                    feature = self.model(X_train)
                    logit = self.cls(feature)
                    loss = F.cross_entropy(logit, y_train) / 10.
                    self.timer.end('train forward')

                    self.timer.begin('train backward')
                    loss.backward()
                    self.timer.end('train backward')

                    err += loss.item()
                    acc += logit.argmax(1).eq(y_train).float().mean()

                backbone_parameters = []
                backbone_parameters.extend(
                    self.model.set_get_trainable_parameters([3, 4]))
                torch.nn.utils.clip_grad.clip_grad_norm_(backbone_parameters + 
                    list(self.cls.parameters()), max_norm=5.0)
                self.opt.step()
                acc /= 10
                LOGGER.info('epoch %2d error: %.6f acc %.6f | time cost - dataload: %.2f forward: %.2f backward: %.2f' % (
                    total_epoch, err, acc,
                    self.timer.query_time_by_name("train data loading", 
                        method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train forward", 
                        method=lambda x:mean(x[-10:])),
                    self.timer.query_time_by_name("train backward", 
                        method=lambda x:mean(x[-10:])),
                ))
            
            # eval loop
            with torch.no_grad():
                self.model.set_mode(False)
                acc_valid = 0
                for i in range (L):

                    x_224, quer_y = valid_task[i]
                    n_sh, n_w = n_shots[i], n_ways[i]

                    x = x_224.to(DEVICE)
                    x = self.model(x)
                    supp_x, quer_x = x[:supp_end[i]], x[supp_end[i]:]

                    supp_x = supp_x.view(n_w, n_sh, supp_x.size(-1))
                    logit = decode_label(supp_x, quer_x).cpu().numpy()
                    acc_valid += (logit.argmax(1) == np.array(quer_y)).mean()
                acc_valid /= len(valid_task)
                LOGGER.info("epoch %2d valid mean acc %.6f" % (total_epoch, 
                    acc_valid))
            
            print('acc_valid: ', acc_valid)
            if best_valid < acc_valid:
                best_param_2nd = best_param
                best_param = pickle.dumps(self.model.state_dict())
                self.model.load_state_dict(pickle.loads(best_param))
                best_valid_2nd = best_valid
                best_valid = acc_valid
                print ('best_valid: ', best_valid)
                cnt = 0
                cnt2 = 0

            else:
                cnt+=1
                cnt2+=1

            if cnt == 3:
                cnt = 0
                if cnt2 == 9: # <<<<<
                    cnt2 = 0
                    self.model.load_state_dict(pickle.loads(best_param_2nd))
                else:
                    self.model.load_state_dict(pickle.loads(best_param)) 

        ###self.model.load_state_dict(pickle.loads(best_param))

        end_time = time.time()
        time_elapsed = (end_time - start_time)
        print("time metatrain = ", time_elapsed)
        return MyLearner(self.model.cpu())


class MyLearner(Learner):

    def __init__(self, model: Wrapper = None) -> None:
        super().__init__()
        self.model = model

    @torch.no_grad()
    def fit(self, support_set: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                               int, int]) -> Predictor:
        self.model.to(DEVICE)
        X_train, y_train, _, n, k = support_set
        X_train, y_train = X_train, y_train
        
        
        return MyPredictor(self.model, X_train, y_train, n, k)

    def save(self, path_to_save: str) -> None:
        torch.save(self.model, os.path.join(path_to_save, "model.pt"))
 
    def load(self, path_to_load: str) -> None:

        if self.model is None:
            self.model = torch.load(os.path.join(path_to_load, 'model.pt'))
    
    
class MyPredictor(Predictor):

    def __init__(self, 
                 model: Wrapper, 
                 supp_x: torch.Tensor, 
                 supp_y: torch.Tensor, 
                 n: int, 
                 k: int) -> None:

        super().__init__()
        self.model = model
        self.other = [supp_x, supp_y, n, k]

    def pipeline_PAg(self, X_train, y_train):

        clf = make_pipeline(MinMaxScaler(), PassiveAggressiveClassifier(max_iter=1000, random_state=0)) # LogisticRegressionCV(max_iter=1000, random_state=0)
        clf.fit(X_train, y_train)
        return clf


    def pipeline_LR(self, X_train, y_train):
        estimators = [
            ('scaler', MinMaxScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]
        pipe = Pipeline(estimators)
        pipe.fit(X_train, y_train)
        return pipe

    @torch.no_grad()
    def predict(self, query_set: torch.Tensor) -> np.ndarray:

        quer_x = query_set
        supp_x, supp_y, n, k = self.other

        ###supp_x = augment(supp_x)

        supp_x = supp_x[supp_y.sort()[1]]
        supp_y = supp_y.sort()[0]


        end = supp_x.size(0)

        x = torch.cat([supp_x, quer_x])
        begin_idx = 0
        XS_224 = []
        t0 = time.time()
        while begin_idx < x.size(0):

            x_128 = x[begin_idx: begin_idx + 128]# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            x_224 = resize_tensor(x_128, 224)
            x_224 = x_224.to(DEVICE)
            XS_224.append(self.model(x_224).cpu())

            begin_idx += 128

        XS_224 = torch.cat(XS_224)
        XS = XS_224  
        print("time cnn = ", time.time() - t0)

        supp_x, quer_x = XS[:end], XS[end:]


        try:
            if k>=3:
                supp_x_sq = np.array(supp_x.view(n, k, supp_x.size(-1)))
                supp_y_sq = np.array(supp_y.view(n, k, 1))
                tr_x, tr_y, val_x, val_y = [], [], [], []
                for _ in range (n):
                    val_x.append(supp_x_sq[_,0])
                    val_y.append(supp_y_sq[_,0])
                    for j in range (k-1):
                        tr_x.append(supp_x_sq[_,1+j])
                        tr_y.append(supp_y_sq[_,1+j])
                tr_x, val_x, tr_y, val_y = np.array(tr_x), np.array(val_x), np.array(tr_y), np.array(val_y)
                #sc =  np.round(accuracy_score(val_y, self.pipeline_LR(tr_x, tr_y).predict(val_x)),3)

                tr_x, val_x = torch.from_numpy(np.float32(tr_x)), torch.from_numpy(np.float32(val_x))
                sc =  np.round(accuracy_score(val_y, np.argmax(decode_label(tr_x.view(n, k-1, tr_x.size(-1)), val_x).cpu().numpy(), axis=1)),3)
                print ('k: ',k)   
                print (sc)

        except:
            print('_____er')


        try:
            x1, x2, x3 = np.mean(np.array(XS)), np.std(np.array(XS)), np.median(np.array(XS))
            print('x1: ',np.round(x1,3), ' x2: ',np.round(x2,3), 'x3: ',np.round(x3,3))
        except:
            x1, x2, x3 = np.mean(np.array(XS.cpu())), np.std(np.array(XS.cpu())), np.median(np.array(XS.cpu()))
            print('x1: ',np.round(x1,3), ' x2: ',np.round(x2,3), 'x3: ',np.round(x3,3))




        #________ Εnsemble of PAg - Gauss ______

        if n <= 2:
            supp_x = supp_x.view(n, k, supp_x.size(-1))
            preds = decode_label(supp_x, quer_x).cpu().numpy()
        elif k <= 3:
            supp_x = supp_x.view(n, k, supp_x.size(-1))
            preds = decode_label(supp_x, quer_x).cpu().numpy()   
        elif n >= 11 and k >= 8 :
                PAg = self.pipeline_PAg(np.array(supp_x), np.array(supp_y)) # fit
                preds = PAg.predict(quer_x)
        elif n < 11 and k < 8 :
                supp_x = supp_x.view(n, k, supp_x.size(-1))
                preds = decode_label(supp_x, quer_x).cpu().numpy()
        else:
            if k >= 3:
                if x3 >= 0.078:
                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds = decode_label(supp_x, quer_x).cpu().numpy()
                elif x1 <= 0.170 and x3 <= 0.020:
                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds = decode_label(supp_x, quer_x).cpu().numpy()
                elif x3  >= 0.047 and x3 <= 0.55 and sc > 0.5 and sc < 0.65:
                    PAg = self.pipeline_PAg(np.array(supp_x), np.array(supp_y)) # fit
                    preds = PAg.predict(quer_x)
                elif sc <= 0.2:
                    PAg = self.pipeline_PAg(np.array(supp_x), np.array(supp_y)) # fit
                    preds = PAg.predict(quer_x)
                elif sc > 0.2 and sc <= 0.50 and x3 >= 0.048 and x3 < 0.070:
                    PAg = self.pipeline_PAg(np.array(supp_x), np.array(supp_y)) # fit
                    preds = PAg.predict(quer_x)
                elif sc >= 0.9 and x1 <= 0.18 and x3 <= 0.04:
                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds = decode_label(supp_x, quer_x).cpu().numpy()
                elif x2 >= 0.48 and x2 <= 0.54 and x3 >= 0.036 and x3 <= 0.040:
                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds = decode_label(supp_x, quer_x).cpu().numpy()
                else:
                    ova_lr = self.pipeline_LR(np.array(supp_x), np.array(supp_y)) # fit
                    preds_ova_lr = ova_lr.predict_proba(quer_x)                          # predict

                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds_gaus = decode_label(supp_x, quer_x).cpu().numpy()

                    preds = np.zeros((preds_ova_lr.shape[0],preds_ova_lr.shape[1]))
                    W = np.argmax(preds_ova_lr,axis=1)
                    Conf = np.zeros((W.shape[0]))
                    for i in range(W.shape[0]):
                            Conf[i] = preds_ova_lr[i, W[i]]
                    mn,st = np.mean(Conf), np.std(Conf)
                    th = mn #- (st*0.1)
                    wlci, wmci = np.argwhere(Conf<=th).reshape(-1,), np.argwhere(Conf>th).reshape(-1,) # do more classes with 0.1*std ....
                    preds[wmci] = preds_ova_lr[wmci]
                    preds[wlci] = preds_gaus[wlci]
            else:
                    supp_x = supp_x.view(n, k, supp_x.size(-1))
                    preds = decode_label(supp_x, quer_x).cpu().numpy()

        return preds
