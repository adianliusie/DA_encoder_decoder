import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from ..train_handler import TrainHandler
from ..helpers import (ConvHandler, DirManager, SSHelper)
from ..models import SystemHandler
from ..utils import (no_grad, toggle_grad, make_optimizer, 
                     make_scheduler)

class BaseLoader(TrainHandler):
    """"base class for running all sequential sentence 
        evaluation and analysis on trained models"""
    
    def __init__(self, exp_name:str, hpc:bool=False):
        self.dir = DirManager.load_dir(exp_name, hpc)
        self.set_up_helpers()
        self.to(self.device)

    def set_up_helpers(self):
        #load training arguments and set up helpers
        self.model_args = self.dir.load_args('model_args')
        super().set_up_helpers(self.model_args)
        
        #load final model
        self.load_model()
        self.model.eval()
    
class EvalHandler(BaseLoader):
    @no_grad
    def evaluate(self, args:namedtuple):
        """ evaluating model performance with loss and accuracy"""
        self.model.eval()
        self.dir.reset_cls_logger()

        #prepare data
        eval_data = self.C.prepare_data(path=args.eval_path, 
                                        lim=args.lim)
        
        eval_batches = self.batcher(data=eval_data, 
                                    bsz=args.bsz, 
                                    shuffle=False)
        
        for k, batch in tqdm(enumerate(eval_batches, start=1)):
            output = self.model_output(batch)
            self.dir.update_cls_logger(output)
            
        loss, acc = self.dir.print_perf(0, None, k, 'test')
        return (loss, acc)
    
    @no_grad
    def predictions(self, eval_data=None, eval_path=None):
        """ output predictions for test set"""
        self.model.eval()
        
        if not eval_data:
            eval_data = self.C.prepare_data(path=eval_path)
        
        eval_batches = self.batcher(data=eval_data, 
                                    bsz=1, shuffle=False)
        
        predictions, labels = [], []
        for k, batch in tqdm(enumerate(eval_batches, start=1)):
            output = self.model_output(batch)
            y = F.softmax(output.logits.squeeze(0), dim=-1)
            predictions.append(y.cpu().numpy())
            labels.append(batch.labels.squeeze(0).cpu().numpy())
        return predictions, labels

class EnsembleEvaluator():
    def __init__(self, exp_name:str):
        seed_names = DirManager.load_ensemble_dir(exp_name)
        self.seeds = (EvalHandler(seed) for seed in seed_names)
        self.C = EvalHandler(seed_names[0]).C
        
    def predictions(self, eval_path):
        eval_data = self.C.prepare_data(path=eval_path)
        
        ensemble = []
        for seed in self.seeds:
            preds, labels = seed.predictions(eval_data)
            ensemble.append(preds)
            
        return ensemble, labels
        
            
            
            
            