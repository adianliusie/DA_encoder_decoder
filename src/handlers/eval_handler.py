import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm
from abc import ABCMeta

from ..train_handler import TrainHandler
from ..helpers import (ConvHandler, DirManager, SSHelper)
from ..models import SystemHandler
from ..utils import (no_grad, toggle_grad, make_optimizer, 
                     make_scheduler)

class BaseLoader(TrainHandler, metaclass=ABCMeta):
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
    
    #util methods to change parellel set up 
    def parallelise(self):
        self.model, self.batcher = SystemHandler.parallelise(
            model=self.model, batcher=self.batcher)
        self.model_args.system = 'whole'
        
    def deparallelise(self):
        self.model, self.batcher = SystemHandler.deparallelise(
            model=self.model, batcher=self.batcher)
        self.model_args.system = 'window'

        
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
    