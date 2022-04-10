import random
from tqdm.notebook import tqdm
import numpy as np
import time

from .conv_handler import Conversation
from ..utils import no_grad, flatten

class SSHelper:
    def __init__(self, C, data, device='cuda'):
        self.C = C
        self.device = device
        self.corpus = data
        self.temp = 0
        
        #split coprora into 5 non-overlapping chunks
        random.shuffle(self.corpus)
        split_len = int(len(data)/5)
        self.splits = (data[i:i+split_len] for i in \
                          range(0, len(data), split_len))
        
        self.utts = set()
        for conv in self.corpus:
            for utt in conv.utts:
                self.utts.add((utt.text, utt.speaker))
        
        self.utts = [{'text':text, 'speaker':speaker} 
                        for text, speaker in list(self.utts)]

    def make_conv(self):
        self.temp += 1
        data = []
        
        start = time.time()
        for conv in next(self.splits):
            context = [self.utt_to_dict(utt) for utt in conv.utts[:-1]]
            if len(context) <= 2: continue
            cur_utt = conv.utts[-1]
            cur_utt = self.utt_to_dict(cur_utt)
            neg_opt = self.make_negative(context, cur_utt)
            
            options = [cur_utt] + neg_opt
            conv_info = {'utterances':context, 
                         'options':options, 
                         'label':0}
            
            data.append(Conversation(conv_info))
            
        synthetic_data = self.C.process_data(data)
        end = time.time()

        print(f'{end-start} seconds taken to get {len(synthetic_data)} synthetic convs')

        return synthetic_data

    @no_grad
    def add_adversary(self, model, bsz=8, N=5000):
        self.model = model
        
        #take a sub sample of the corpus 
        random.shuffle(self.utts)
        
        self.utt_texts = self.utts[:N]
        utts = [i['text'] for i in self.utts[:N]]
        
        #generate vectors for all utterances
        #utts = [i['text'] for i in self.utts]
        
        utt_vecs = []
        batches = [utts[i:i+bsz] for i in range(0,len(utts), bsz)]
        
        start = time.time()
        for batch in batches:
            model_input = self.C.tokenizer(batch, padding=True, 
                                           return_tensors='pt').to(self.device)
            y = model.utt_encode(**model_input)
            utt_vecs += y.tolist()
        
        self.utt_vecs = np.array(utt_vecs)  #[|U|, 768]
        end = time.time()
        print(f'{end-start} seconds taken to get {len(self.utt_vecs)} utt vectors')
              
    @no_grad 
    def get_kneighbors(self, context, n): 
        context_text = ' '.join([utt['text'] for utt in context])
        text_ids = self.C.tokenizer([context_text], return_tensors='pt').to(self.device)
        
        utt_vec = self.model.utt_encode(**text_ids)  #[1, 768]
        utt_vec = utt_vec.cpu().numpy()

        neg_scores = np.dot(self.utt_vecs, utt_vec.T)  #[|U|, 1]
        neg_scores = neg_scores.squeeze(-1)
        neg_ind = np.argpartition(neg_scores, -1*n)[-1*n:]
        neg_ind = [int(i) for i in neg_ind]
        neg_utts = [self.utt_texts[i] for i in neg_ind]
        return neg_utts
        
    def make_negative(self, context, cur_utt, n=3):
        if hasattr(self, 'utt_vecs') and self.temp != 3:
            neg_utts = self.get_kneighbors(context, n+1)
            neg_utts = [utt for utt in neg_utts if utt['text'] != cur_utt['text']]
            neg_utts = neg_utts[:n]
        else:
            #random selection
            neg_utts = []
            while len(neg_utts) < n:
                rand_utt = self.rand_utt()
                if rand_utt['text'] != cur_utt['text']:
                    neg_utts.append(rand_utt)
        return neg_utts
        
    @staticmethod
    def utt_to_dict(utt):
        return {'text':utt.text, 'speaker':utt.speaker}
    
    def rand_utt(self):
        return random.choice(self.utts)
