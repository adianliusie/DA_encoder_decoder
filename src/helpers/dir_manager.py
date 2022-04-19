import os
import json
import torch
import shutil
import csv
import numpy as np

from types import SimpleNamespace
from typing import Callable
from collections import namedtuple

from ..utils import load_json, download_hpc_model
from ..config import config

BASE_DIR = f'{config.base_dir}/trained_models'

class DirManager:
    """ Class which manages logs, models and config files """

    ### Methods for Initialisation of Object ############################
    
    def __init__(self, exp_name:str=None, temp:bool=False):
        if temp:
            print("using temp directory")
            self.exp_name = 'temp'
            self.del_temp_dir()
        else:
            self.exp_name = exp_name

        self.make_dir()
        self.log = self.make_logger(file_name='log')
    
    def del_temp_dir(self):
        """deletes the temp, unsaved experiments directory"""
        if os.path.isdir(f'{BASE_DIR}/temp'): 
            shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        """makes experiments directory"""
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def make_logger(self, file_name:str)->Callable:
        """creates logging function which saves prints to txt file"""
        log_path = f'{self.path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            print(*x)    
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    ### Methods for Logging performance ###############################

    def reset_cls_logger(self):
        self.cum_loss  = 0
        self.cum_hits  = 0
        self.cum_preds = 0

    def update_cls_logger(self, loss=0, hits=0, num_preds=0):
        self.cum_loss  += loss
        self.cum_hits  += hits
        self.cum_preds += num_preds
        
    def print_perf(self, epoch:int, k:int, print_len:int, mode='train'):
        """returns and logs performance"""
        loss = f'{self.cum_loss/print_len:.3f}'
        acc  = f'{self.cum_hits/self.cum_preds:.3f}' if self.cum_preds else 0
        
        if mode == 'train':
            self.update_curve(mode, epoch, float(loss), float(acc))
            self.log(f'{epoch:<3} {k:<5}   loss {loss}   acc {acc}')
        elif mode == 'dev':
            self.update_curve(mode, epoch, float(loss), float(acc))
            self.log(f'{epoch:<3} DEV     loss {loss}   acc {acc}')
        elif mode == 'test':
            self.log(f'{epoch:<3} TEST    loss {loss}   acc {acc}')
        self.reset_cls_logger()
        return float(loss), float(acc) 
                   
    def update_curve(self, mode, *args):
        """ logs any passed arguments into a file"""
        with open(f'{self.path}/{mode}.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(args)

    ### Utility Methods ################################################
    
    @property
    def path(self):
        """returns base experiment path"""
        return f'{BASE_DIR}/{self.exp_name}'

    def file_exists(self, file_name:str):
        return os.path.isfile(f'{self.path}/{file_name}.json') 
    
    def save_args(self, name:str, args:namedtuple):
        """saves arguments into json format"""
        config_path = f'{self.path}/{name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(args.__dict__, jsonFile, indent=4)

    def save_dict(self, name:str, dict_data):
        save_path = f'{self.path}/{name}.json'
        with open(save_path, 'x') as jsonFile:
            json.dump(dict_data, jsonFile, indent=4)
  
    ### Methods for loading existing dir ##############################
    
    @classmethod
    def load_dir(cls, exp_name:str, hpc=False)->'DirManager':
        dir_manager = cls.__new__(cls)
        if hpc: 
            dir_manager.exp_name = 'hpc/'+exp_name
            download_hpc_model(exp_name)
        else:
            dir_manager.exp_name = exp_name
        
        dir_manager.log = print
        return dir_manager
    
    @classmethod
    def load_ensemble_dir(cls, exp_name:str, hpc=False)->'DirManager':
        base_dir = f'{BASE_DIR}/{exp_name}'
        
        seeds = []
        for seed in os.listdir(base_dir):
            seeds.append(f'{exp_name}/{seed}')
        return seeds
    
    def load_args(self, name:str)->SimpleNamespace:
        args = load_json(f'{self.path}/{name}.json')
        return SimpleNamespace(**args)
    
    def load_dict(self, name:str)->dict:
        return load_json(f'{self.path}/{name}.json')
    
    def load_curve(self, mode='train'):
        float_list = lambda x: [float(i) for i in x] 
        with open(f'{self.path}/{mode}.csv') as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            data_read = [float_list(row) for row in reader]
        return tuple(zip(*data_read))
    
