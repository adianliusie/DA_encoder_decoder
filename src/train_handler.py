import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from types import SimpleNamespace
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm.notebook import tqdm

from .helpers import (ConvHandler, DirManager, SSHelper)
from .models import SystemHandler
from .utils import (no_grad, toggle_grad, make_optimizer, 
                    make_scheduler)


class TrainHandler:
    """"base class for running all sequential sentence/utterance 
        classification experiments"""
    
    def __init__(self, exp_name, args:namedtuple):
        self.dir = DirManager(exp_name, args.temp)
        self.dir.save_args('model_args', args)

        self.set_up_helpers(args)
        
    def set_up_helpers(self, args): 
        self.model_args = args
        self.system_args = args.system_args
        self.max_len = args.max_len
        
        special_tokens = []     
        if (args.formatting == 'spkr_sep'):
            special_tokens += ['[SPKR_1]', '[SPKR_2]']
        
        self.C = ConvHandler(transformer=args.transformer, 
                             filters=args.filters, 
                             special_tokens=special_tokens)

        self.batcher = SystemHandler.batcher(
                           formatting=args.formatting,
                           system=args.encoder,
                           max_len=args.max_len)
        
        self.model = SystemHandler.make_seq2seq(
                        transformer=args.transformer,
                        encoder=args.encoder, 
                        decoder=args.decoder, 
                        system_args=args.system_args,
                        num_labels=args.num_labels,                         
                        C=self.C)
        
        self.device = args.device

    def set_up_data_filtered(self, paths, lim:int)->list:
        data = [self.C.prep_filtered_data(path=path, max_len=self.max_len, 
                             lim=lim) if path else None for path in paths]
        return data
    
    def set_up_data(self, paths, lim:int)->list:
        data = [self.C.prepare_data(path=path, max_len=self.max_len, 
                       lim=lim) if path else None for path in paths]
        return data
    
    def set_up_opt(self, args:namedtuple):
        optimizer = make_optimizer(opt_name=args.optim, 
                                   lr=args.lr, 
                                   params=self.model.parameters())
        
        if args.sched:
            steps = (len(train)*args.epochs)/args.bsz
            scheduler = make_scheduler(optimizer=optimizer, 
                                       steps=steps,
                                       mode=args.sched)
        else:
            scheduler = None
            
        return optimizer, scheduler 
    
    ######  Methods For Dialogue Act Classification  ##########
    
    def train(self, args:namedtuple):
        self.dir.save_args('train_args', args)
        self.to(self.device)

        paths = [args.train_path, args.dev_path, args.test_path]
 
        train, dev, test = self.set_up_data_filtered(paths, args.lim)
        #train, dev, test = self.set_up_data(paths, args)
        optimizer, scheduler = self.set_up_opt(args)
        
        best_epoch = (-1, 10000, 0)
        for epoch in range(args.epochs):
            self.model.train()
            self.dir.reset_cls_logger()
            train_b = self.batcher(data=train, bsz=args.bsz, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
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
                if k%args.print_len == 0:
                    self.dir.print_perf(epoch, k, args.print_len, 'train')
            
            if not args.dev_path:
                self.save_model()
            else:
                self.model.eval()
                self.dir.reset_cls_logger()
                
                dev_b = self.batcher(data=dev, bsz=args.bsz, shuffle=True)
                for k, batch in enumerate(dev_b, start=1):
                    output = self.model_output(batch, no_grad=True)
                    self.dir.update_cls_logger(output)
                   
                # print dev performance 
                loss, acc = self.dir.print_perf(epoch, None, k, 'dev')

                # save performance if best dev performance 
                if acc > best_epoch[2]:
                    self.save_model()
                    best_epoch = (epoch, loss, acc)

            if args.test_path:
                self.dir.reset_cls_logger()
                test_b = self.batcher(data=test, bsz=args.bsz, shuffle=True)
                for k, batch in enumerate(test_b, start=1):
                    output = self.model_output(batch, no_grad=True)
                    self.dir.update_cls_logger(output)
                loss, acc = self.dir.print_perf(epoch, None, k, 'test')

            #update scheduler if step with each epoch
            if args.sched == 'step': 
                scheduler.step()
                       
        self.dir.log(f'epoch {best_epoch[0]}  loss: {best_epoch[1]:.3f} ',
                     f'acc: {best_epoch[2]:.3f}')
    
        self.load_model()

    @toggle_grad
    def model_output(self, batch):
        """flexible method for dealing with different set ups. 
           Returns loss and accuracy statistics"""
        
        trans_inputs = {'input_ids':batch.ids, 
                        'attention_mask':batch.mask}
        
        #add extra transformer arguments if necessary
        if self.model_args.system_args:
            if 'token_type_embed' in self.model_args.system_args:
                trans_inputs['token_type_ids'] = batch.spkr_ids
            if 'spkr_embed' in self.model_args.system_args:
                trans_inputs['speaker_ids'] = batch.spkr_ids
            if 'utt_embed' in self.model_args.system_args:
                trans_inputs['utterance_ids'] = batch.utt_ids     

        #add model dependent arguments
        system_inputs = {}
        system_inputs['utt_pos'] = batch.utt_pos
        if self.model_args.decoder in ['transformer', 'rnn']:
            system_inputs['labels'] = batch.labels
        
        y = self.model(trans_inputs, **system_inputs)
        
        if len(batch.labels.shape) == 2:
            loss = F.cross_entropy(y.view(-1, y.shape[-1]), 
                                   batch.labels.view(-1))
        else:  
            loss = F.cross_entropy(y, batch.labels)
        
        hits = torch.argmax(y, dim=-1) == batch.labels
        hits = torch.sum(hits[batch.labels != -100]).item()
        num_preds = torch.sum(batch.labels != -100).item()
        return SimpleNamespace(loss=loss, logits=y,
                               hits=hits, num_preds=num_preds)

    #############   SAVING AND LOADING    #############
    
    def save_model(self, name='base'):
        device = next(self.model.parameters()).device
        self.model.to("cpu")
        torch.save(self.model.state_dict(), 
                   f'{self.dir.path}/models/{name}.pt')
        self.model.to(self.device)

    def load_model(self, name='base'):
        self.model.load_state_dict(
            torch.load(self.dir.path + f'/models/{name}.pt'))

    def to(self, device):
        if hasattr(self, 'model'):   self.model.to(device)
        if hasattr(self, 'batcher'): self.batcher.to(device)

