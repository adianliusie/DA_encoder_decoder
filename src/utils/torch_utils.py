import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR

from typing import Callable

######## GRADIENT DECORATORS  ###########
def no_grad(func:Callable):
    """decorator which detaches gradients"""
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner

def toggle_grad(func:Callable):
    """decorator which lets one toggle gradients with argument"""
    def inner(*args, no_grad:bool=False):
        if no_grad==True:
            with torch.no_grad():
                return func(*args)
        else:
            return func(*args)
    return inner

#############   OPTIMIZERS  ############### 
def make_optimizer(opt_name:str, params, lr:float):
    if opt_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr)
    return optimizer 

#############   SCHEDULERS  ############### 
def triangle(SGD_steps, warm_up=0.1)->Callable:
    print(SGD_steps, warm_up)
    def inner_func(i):
        if i <= SGD_steps*warm_up:
            output = i/(SGD_steps*warm_up)
        else:
            output = 1-((i-warm_up*SGD_steps)/(SGD_steps*(1-warm_up)))
            output = max(output, 0)
        return output
    return inner_func

def make_scheduler(optimizer, mode:str, steps:int=None, **kwargs):
    if mode == 'triangular':
        scheduler = LambdaLR(optimizer, lr_lambda=triangle(steps))
    elif mode == 'step':
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    return scheduler 


## don't remember when I wrote this, I don't need it, but it's really smart
def context_mask(past=0, future=0):
    lower = torch.tril(torch.ones([x,x]), diagonal=future)
    upper = torch.triu(torch.ones([x,x]), diagonal=-past)
    return lower*upper
