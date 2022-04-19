import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List, Tuple
from types import SimpleNamespace
import random
import numpy as np

from ..utils import flatten
from abc import ABCMeta

class BaseBatcher(metaclass=ABCMeta):
    """base class that creates batches for training/eval for all tasks"""

    def __init__(self, formatting:str=None, max_len:int=None, **kwargs):
        """initialises object"""
        self.device = torch.device('cpu')
        self.max_len = max_len
        self.formatting = formatting
        
    def batches(self, data:List['Conversations'], 
                      bsz:int, shuffle:bool=False):
        convs = self._prep_convs(data)
        if shuffle: random.shuffle(convs)
        batches = [convs[i:i+bsz] for i in range(0,len(convs), bsz)]
        for batch in batches:
            yield self.batchify(batch)
        #batches = [self.batchify(batch) for batch in batches]       
        #return batches
    
    def _get_padded_ids(self, ids:list)->("padded ids", "padded_mask"):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [0]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask
    
    def _pad_seq(self, x:list, pad_val:int=0)->list:
        """pads input so can be put in a tensor"""
        max_len = max([len(i) for i in x])
        x_pad = [i + [pad_val]*(max_len-len(i)) for i in x]
        x_pad = torch.LongTensor(x_pad).to(self.device)
        return x_pad
       
    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
         
    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
    
    
class FlatBatcher(BaseBatcher):
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkr_ids, utt_ids, utt_pos_seq, labels = zip(*batch)  
        ids, mask = self._get_padded_ids(ids)
        spkr_ids = self._pad_seq(spkr_ids)
        utt_ids = self._pad_seq(utt_ids)
        
        utt_pos_seq = [torch.LongTensor(utt).to(self.device) for utt in utt_pos_seq] #[bsz, 1]
        labels = self._pad_seq(labels, pad_val=-100)
                
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                   spkr_ids=spkr_ids, utt_ids=utt_ids, utt_pos=utt_pos_seq)
    
    def _prep_convs(self, data:List['Conversations']):
        """ sequence classification input data preparation"""
        output = []
        for conv in data:
            #get all utterances in conv and labels
            ids = [utt.ids for utt in conv.utts]
            spkrs = [utt.spkr_id[0] for utt in conv.utts]
            spkrs_tok = [utt.spkr_id[1] for utt in conv.utts]
            ids, utt_pos_seq = self._format_ids(ids, spkrs_tok)

            #get utterance meta information
            spkr_ids = [[s]*len(i) for s, i in zip(spkrs, ids)]
            spkr_ids = flatten(spkr_ids)
            utt_ids = [[k]*len(i) for k, i in enumerate(ids)]
            utt_ids = flatten(utt_ids)
            ids = flatten(ids)
            
            labels = [utt.label for utt in conv]
                      
            #add to data set    
            if self.max_len==None or len(utt_ids)<self.max_len:
                output.append([ids, spkr_ids, utt_ids, utt_pos_seq, labels])
                
        return output
    
    def _format_ids(self, utts, spkrs_tok):
        CLS, SEP = utts[0][0], utts[0][-1]
        utt_pos_seq = None  #position of all utt special tokens
        
        # [CLS] U1 [SEP] U2 [SEP] ... [SEP] UN [SEP] 
        if not self.formatting:
            utt_ids = [utt[1:] for utt in utts]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_pos_seq = np.cumsum([len(utt) for utt in utt_ids])-1

        
        # [CLS] [A] U1 [B] U2 ... [A] UN [SEP] 
        elif self.formatting == 'spkr_sep':
            utt_ids = [[s] + utt[1:-1] for utt, s in zip(utts, spkrs_tok)]
            utt_ids[0] = [CLS] + utt_ids[0]
            utt_ids[-1] = utt_ids[-1] + [SEP]
            
            utt_pos_seq = np.cumsum([len(utt) for utt in utt_ids])
            utt_pos_seq = [1] + list(utt_pos_seq[:-1])
            print('untested batching (batchers, _format_ids)')
            
        else:
            raise ValueError('invalid sequence formatting')
        return utt_ids, utt_pos_seq

class HierBatcher(BaseBatcher):
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        ids, spkrs, labels = zip(*batch)  
        
        # save conv start and end positions as about to be flattened
        conv_lens = [len(conv) for conv in ids]
        cum_lens = np.cumsum([0] + conv_lens)
        conv_splits = [(cum_lens[i], cum_lens[i+1]) \
                              for i in range(len(conv_lens))]
        
        # flatten inputs to 2D tensor for parallel processing
        flat_ids = flatten(ids)
        ids, mask = self._get_padded_ids(flat_ids)
        
        #labels are returned in full dim
        labels = self._pad_seq(labels, pad_val=-100)
                
        return SimpleNamespace(ids=ids, mask=mask, labels=labels, 
                   spkr_ids=spkrs, conv_splits=conv_splits)
    
    def _prep_convs(self, data:List['Conversations']):
        """ sequence classification input data preparation"""
        output = []
        for conv in data:
            #get all utterances in conv and labels
            ids = [utt.ids for utt in conv.utts]
            spkrs = [utt.spkr_id[0] for utt in conv.utts]
            labels = [utt.label for utt in conv]
                 
            #add to data set    
            #max_utt_len = max([len(i) for i in ids])
            if self.max_len==None or len(ids)<self.max_len:
                output.append([ids, spkrs, labels])
                
        return output
    
