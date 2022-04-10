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

class SSHandler(TrainHandler)
    def SS_next_sentence(self, args:namedtuple):
        self.dir.save_args('SS_args', args)

        batcher, optimizer, scheduler = self.set_up_opt(args)
        corpus = self.C.prepare_data(path=args.train_path, 
                                     lim=args.lim)

        SS_helper = SSHelper(self.C, corpus, device='cuda')
        
        for epoch in range(args.epochs):
            logger = np.zeros(3)
            self.model.train()
            
            train = SS_helper.make_conv()
            train_batches = batcher(data=train, 
                                    bsz=args.bsz, 
                                    shuffle=True)
            
            for k, batch in enumerate(train_batches, start=1):
                #forward and loss calculation
                output = self.model_output(batch)
                loss = output.loss

                #updating model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #update scheduler if step with each batch
                if args.sched == 'triangular': 
                    scheduler.step()

                #accuracy logging
                logger += [output.loss.item(), 
                           output.hits, 
                           output.num_preds]

                #print every now and then
                if k%args.print_len == 0:
                    loss = f'{logger[0]/args.print_len:.3f}'
                    acc  = f'{logger[1]/logger[2]:.3f}'
                    self.dir.update_curve('SS_train', epoch, loss, acc)
                    self.dir.log(f'{epoch:<3} {k:<5}  ',
                                 f'loss {loss}   acc {acc}')
                    logger = np.zeros(3)
            
            SS_helper.add_adversary(self.model, bsz=4)
           