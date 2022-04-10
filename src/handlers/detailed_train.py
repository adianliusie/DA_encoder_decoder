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

class DetailedTrain(TrainHandler):
    def train(self, args:namedtuple):
        self.dir.save_args('train_args', args)
        self.to(self.device)

        paths = [args.train_path, args.dev_path, args.test_path]
 
        train, dev, test = self.set_up_data_filtered(paths, args.lim)
        optimizer, scheduler = self.set_up_opt(args)
        
        k=0
        best_state = (-1, 10000, 0)
        for epoch in range(args.epochs):
            self.model.train()
            self.dir.reset_cls_logger()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for batch in train_b:
                #forward and loss calculation
                output = self.model_output(batch)
                loss = output.loss
                
                #updating model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #update scheduler if step with each batch
                if args.sched == 'triangular': scheduler.step()

                #accuracy logging
                self.dir.update_cls_logger(output)
                
                #print train performance every now and then
                k+=1
                if k%args.print_len == 0:
                    self.dir.print_perf(epoch, k, args.print_len, 'train')
            
                #every 200 SGD steps, assess performance on DEV
                if k%500 == 0:
                    if args.dev_path:
                        self.model.eval()
                        self.dir.reset_cls_logger()

                        dev_b = self.batcher(data=dev, bsz=args.bsz, shuffle=True)
                        for i, batch in enumerate(dev_b, start=1):
                            output = self.model_output(batch, no_grad=True)
                            self.dir.update_cls_logger(output)

                        # print dev performance 
                        loss, acc = self.dir.print_perf(epoch, None, i, 'dev')

                        # save performance if best dev performance 
                        if acc > best_state[2]:
                            self.save_model()
                            best_state = (epoch, loss, acc)
                        
            #update scheduler if step with each epoch
            if args.sched == 'step': 
                scheduler.step()
                       
        self.dir.log(f'epoch {best_state[0]}  loss: {best_state[1]:.3f} ',
                     f'acc: {best_state[2]:.3f}')
    
        self.load_model()

           
