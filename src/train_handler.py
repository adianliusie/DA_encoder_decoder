import wandb
import numpy as np
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import namedtuple
from types import SimpleNamespace
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
        
        self.C = ConvHandler(system=args.system, 
                             filters=args.filters, 
                             special_tokens=special_tokens)

        self.batcher = SystemHandler.batcher(
                           formatting=args.formatting,
                           encoder=args.encoder,
                           max_len=args.max_len)
        
        self.model = SystemHandler.make_seq2seq(
                        system=args.system,
                        encoder=args.encoder, 
                        decoder=args.decoder, 
                        system_args=args.system_args,
                        num_labels=args.num_labels,                         
                        C=self.C)
        
        self.device = args.device
    
    def set_up_data(self, paths, lim:int)->list:
        data = [self.C.prepare_data(path=path, lim=lim)
                   if path else None for path in paths]
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
        
        ### INITIALISE WANDB
        if args.wandb:
            wandb.init(project=f'SWDA-{args.wandb}', entity="adian", 
                       name=self.dir.exp_name, reinit=True)
            
            ### SAVE EXPEIRMENT ARGS FOR WANDB
            cfg = {k:v for k, v in vars(args).items() if k in
                               ['bsz', 'epochs', 'sched', 'lr', 'optim', 'layer']}
            cfg['encoder'] = self.model_args.encoder
            cfg['decoder'] = self.model_args.decoder
            
            if self.model_args.system_args:
                num_layers =  [i for i in self.model_args.system_args if 'layers' in i]
                rand_layers = [i for i in self.model_args.system_args if 'rand' in i]
                cfg['layers']      = next(iter(num_layers),  None)
                cfg['rand_layers'] = next(iter(rand_layers), None)
                
            wandb.config.update(cfg) 
            
            #### SET EVAL METRICS
            wandb.define_metric("dev_loss", step_metric="epoch" ,summary="min")
            wandb.define_metric("dev_acc",  step_metric="epoch", summary="max")
            wandb.watch(self.model)

        self.dir.save_args('train_args', args)
        self.to(self.device)

        paths = [args.train_path, args.dev_path, args.test_path]
 
        train, dev, test = self.set_up_data(paths, args.lim)
        optimizer, scheduler = self.set_up_opt(args)
        
        best_epoch = (-1, 10000, 0)
        for epoch in range(args.epochs):
            self.model.train()
            self.dir.reset_cls_logger()
            train_b = self.batcher(data=train, bsz=1, shuffle=True)
            
            for k, batch in enumerate(train_b, start=1):
                output = self.model_output(batch)
                loss = output.loss/args.bsz
                loss.backward()
                
                ### SYNTHETICALLY ALLOWS LARGE BATCH SIZES
                if k%args.bsz==0:
                    optimizer.step()
                    optimizer.zero_grad()

                #update scheduler if step with each batch
                if args.sched == 'triangular': scheduler.step()

                #accuracy logging
                self.dir.update_cls_logger(loss=output.loss, 
                                           hits=output.hits, 
                                           num_preds=output.num_preds)

                #print train performance every now and then
                if k%args.print_len == 0:
                    loss, acc = self.dir.print_perf(epoch, k, args.print_len, 'train')
                    if args.wandb: wandb.log({"epoch":epoch, "loss":loss, "acc":acc})

            if not args.dev_path:
                if args.save: self.save_model()
            else:
                self.model.eval()
                self.dir.reset_cls_logger()
                
                dev_b = self.batcher(data=dev, bsz=1, shuffle=True)
                for k, batch in enumerate(dev_b, start=1):
                    output = self.model_output(batch, decode=True, no_grad=True)
                    self.dir.update_cls_logger(loss=output.loss, 
                                               hits=output.hits, 
                                               num_preds=output.num_preds)
                   
                # print dev performance 
                loss, acc = self.dir.print_perf(epoch, None, k, 'dev')
                if args.wandb: wandb.log({"dev_loss":loss, "dev_acc":acc})

                # save performance if best dev performance 
                if acc > best_epoch[2]:
                    if args.save: self.save_model()
                    best_epoch = (epoch, loss, acc)

            if args.test_path:
                self.dir.reset_cls_logger()
                test_b = self.batcher(data=test, bsz=1, shuffle=True)
                for k, batch in enumerate(test_b, start=1):
                    output = self.model_output(batch, decode=True, no_grad=True)
                    self.dir.update_cls_logger(loss=output.loss, 
                                               hits=output.hits, 
                                               num_preds=output.num_preds)
                loss, acc = self.dir.print_perf(epoch, None, k, 'test')

            #update scheduler if step with each epoch
            if args.sched == 'step': 
                scheduler.step()
                       
        self.dir.log(f'epoch {best_epoch[0]}  loss: {best_epoch[1]:.3f} ',
                     f'acc: {best_epoch[2]:.3f}')
    
        self.load_model()

    @toggle_grad
    def model_output(self, batch, decode=False):
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
        if self.model_args.encoder in ['utt_trans']:
            system_inputs['utt_pos'] = batch.utt_pos
        elif self.model_args.encoder in ['hier']:
            system_inputs['conv_splits'] = batch.conv_splits
            
        #forward of the model in training
        if (not decode) or (self.model_args.decoder=='linear'):
            if self.model_args.decoder in ['transformer', 'rnn', 'crf']:
                #add labels for autoreressive decoders
                system_inputs['labels'] = batch.labels
        
            
            if self.model_args.decoder == 'crf':
                #CRF only returns the loss
                loss = self.model(trans_inputs, **system_inputs)
                y, hits, num_preds = 0, 0, 0
            else:
                #other models return logits
                y = self.model(trans_inputs, **system_inputs)

                #calculate cross entropy loss
                if len(batch.labels.shape) == 2:
                    loss = F.cross_entropy(y.view(-1, y.shape[-1]), batch.labels.view(-1))
                else:  
                    loss = F.cross_entropy(y, batch.labels)

                #return accuracy metrics
                hits = torch.argmax(y, dim=-1) == batch.labels
                hits = torch.sum(hits[batch.labels != -100]).item()
                num_preds = torch.sum(batch.labels != -100).item()
                
        #evaluation decoding (does not use the given labels)
        elif (decode and self.model_args.decoder=='crf'):
            preds = self.model.decode(trans_inputs, **system_inputs)
            preds = torch.LongTensor(preds).to(self.device)
            hits = (preds == batch.labels)
            hits = torch.sum(hits[batch.labels != -100]).item()
            num_preds = torch.sum(batch.labels != -100).item()
            y, loss = 0, 0
            
        return SimpleNamespace(loss=loss, y=y,
                               hits=hits, num_preds=num_preds)

    #############   MODEL UTILS      ##################
    def load_encoder(self, path, freeze=False):
        self.model = SystemHandler.load_encoder(self.model, path, freeze=True)
    
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

