from types import SimpleNamespace
from typing import List
from tqdm import tqdm 

import json
import re 
import os

from ..config import config
from ..utils import load_json, flatten, load_list 
from ..models.hugging_utils import get_tokenizer

class Utterance():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f'{self.speaker}: {self.text}'
    
class Conversation():
    def __init__(self, data:dict):
        for k, v in data.items():
            if k == 'utterances':
                self.utts = [Utterance(**utt) for utt in v]
            elif k == 'options':
                self.options = [Utterance(**opt) for opt in v]
            else:
                setattr(self, k, v)
                
    def __iter__(self):
        return iter(self.utts)

    def __getitem__(self, k):
        return self.utts[k]

class ConvHandler:    
    def __init__(self, transformer, filters=None, 
                       special_tokens=None, tqdm_disable=False):
        """ Initialises the Conversation helper """
        
        if transformer:
            self.transformer = transformer
            self.tokenizer = get_tokenizer(transformer)
        
        if special_tokens:
            self.add_speaker_tokens(special_tokens)
            
        self.cleaner = TextCleaner(filters=filters)
        self.tqdm_disable = tqdm_disable
    
    def add_speaker_tokens(self, tokens):
        self.tokenizer.add_tokens(tokens, special_tokens=True)
        
    def prep_filtered_data(self, path, lim=None, max_len=4090, quiet=False):
        def conv_len(conv):
            utt_tok_len = [len(utt.ids[:-1]) for utt in conv]
            return 1 + sum(utt_tok_len)
        
        data = self.prepare_data(path, lim, quiet)
        data = [conv for conv in data if conv_len(conv)<=max_len]
        return data
    
    def prepare_data(self, path:str, lim:int=None, quiet=False)->List:
        """ Given path, will load json and process data for downstream tasks """
        assert path.split('.')[-1] == 'json', "data must be in json format"

        path = f'{config.base_dir}/data/{path}'
        raw_data = load_json(path)
        self.load_label_info(path)

        data = [Conversation(conv) for conv in raw_data]
        data = self.process_data(data, lim, quiet)
        return data
    
    def process_data(self, data, lim=None, quiet=False):
        if lim: data = data[:lim]
        self.clean_text(data)
        self.tok_convs(data, quiet)  
        self.get_speaker_ids(data)
        if hasattr(self, 'label_dict'): self.get_label_names(data)
        return data
    
    def load_label_info(self, path):
        if not hasattr(self, 'label_dict'):
            #replace filename before extension with `labels'
            label_path = re.sub(r'\/(\w*?)\.', '/labels.', path)
            if os.path.isfile(label_path): 
                label_dict = load_json(label_path)
                self.label_dict = {int(k):v for k, v in label_dict.items()}
               
    def clean_text(self, data:List[Conversation]):
        """ processes text depending on arguments. E.g. punct=True filters
        punctuation, action=True filters actions etc."""
        for conv in data:
            for utt in conv:
                utt.text = self.cleaner.clean_text(utt.text)
    
    def tok_convs(self, data:List[Conversation], quiet=False):
        """ generates tokenized ids for each utterance in Conversation """
        for conv in tqdm(data, disable=quiet):
            for utt in conv.utts:
                utt.ids = self.tokenizer(utt.text).input_ids
            
    def get_label_names(self, data:List[Conversation]):
        """ generates detailed label name for each utterance """
        for conv in data:
            for utt in conv:
                utt.label_name = self.label_dict[utt.label]
                
    def get_speaker_ids(self, data:List[Conversation]):
        speakers = set([utt.speaker for conv in data for utt in conv])
        speakers = sorted(list(speakers))
        
        self.speaker_dict = {s:k for k, s in enumerate(speakers)}

        for conv in data:
            for utt in conv:
                speaker_id = self.speaker_dict[utt.speaker]
                tok = self.tokenizer(f'[SPKR_{speaker_id}]').input_ids[1]
                utt.spkr_id = (speaker_id, tok)
    
    def __getitem__(self, x:str):
        """ returns conv with a given conv_id if exists in dataset """
        for conv in self.data:
            if conv.conv_id == str(x): return conv
        raise ValueError('conversation not found')
             
    def __contains__(self, x:str):
        """ checks if conv_id exists in dataset """
        output = False
        if x in [conv.conv_id for conv in self.data]:
            output = True
        return output
       

class TextCleaner:
    def __init__(self, filters=None):
        if filters:
            self.punct = 'punctuation' in filters
            self.action = 'action' in filters
            self.hes = 'hesitation' in filters  
        else:
            self.punct = False
            self.action = False
            self.hes = False
            
    def clean_text(self, text:str)->str:
        """method which cleans text with chosen convention"""
        if self.action:
            text = re.sub("[\[\(\<\%].*?[\]\)\>\%]", "", text)    
        if self.punct: 
            text = re.sub(r'[^\w\s]', '', text)
            text = text.lower()
        if self.hes:
            text = self.hesitation(text)
        text = ' '.join(text.split())
        return text

    @staticmethod
    def hesitation(text:str)->str:
        """internal function to converts hesitation"""
        hes_maps = {"umhum":"um", "uh-huh":"um", 
                    "uhhuh":"um", "hum":"um", "uh":'um'}

        for h1, h2 in hes_maps.items():
            if h1 in text:
                pattern = r'(^|[^a-zA-z])'+h1+r'($|[^a-zA-Z])'
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
                #run line twice as uh uh share middle character
                text = re.sub(pattern, r'\1'+h2+r'\2', text)
        return text 
