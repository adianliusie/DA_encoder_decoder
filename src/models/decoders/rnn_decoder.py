import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
from types import SimpleNamespace

# code based loosely on the following 2 projects
# https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
# https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/DecoderRNN.py

class BeamDecoder:          
    def decode(self, encoder_outputs, encoder_mask, beam_width=5):
        output = []
        for idx, utt_embeds in enumerate(encoder_outputs): 
            # set up decoding, init queue and set start variables
            device = utt_embeds.device
            cur_utt = utt_embeds[0]
            hx = torch.zeros([self.h_size], device=device)
            cx = torch.zeros([self.h_size], device=device)
            
            # add first point to states
            first_node = SimpleNamespace(path=[self.start_tok], log_prob=0, h=hx, c=cx)
            current_states = [first_node]    
            
            # do beam search
            for utt_num in range(len(utt_embeds)):
                utt = utt_embeds[utt_num]
                next_states = []
                for node in current_states:
                    prev_label =  torch.LongTensor([node.path[-1]]).to(device)
                    
                    # run through next cell and select top k probabilities
                    y, hx, cx = self.step(embed=utt, label=prev_label, hx=node.h, cx=node.c)
                    y_log_probs = F.log_softmax(y, dim=-1)
                    log_prob, indexes = torch.topk(y_log_probs, beam_width)
                    
                    # add all new states to next search space
                    for prob, ind in zip(log_prob, indexes):
                        path = node.path.copy() + [ind.item()]
                        prob = node.log_prob + prob
                        next_node = SimpleNamespace(path=path, log_prob=prob, h=hx, c=cx)
                        next_states.append(next_node)
                    
                # prune states to the best k states
                next_states.sort(key=lambda x: x.log_prob, reverse=True)
                next_states = next_states[:beam_width]
                current_states = next_states
            
            solution = current_states[0]
            output.append(solution.path[1:])
        output = torch.LongTensor(output).to(device) #[B, N]
        output = F.one_hot(output, num_classes=self.num_class).float() #[B, N, C]
        return output
    
class RNNDecoder(nn.Module, BeamDecoder):
    def __init__(self, num_class, cell_type='lstm', embed_size=10, h_size=768, rnn_h_size=768, dropout=0.0):
        '''RNN decoder which when given a sequence of vectors, outputs sequence of decisions.
           Only for sequence labelling tasks (i.e. x_{1:N} -> y_{1:N}) and does not learn stopping'''
        super().__init__()
        
        #make embeddings for labels
        self.embedding = nn.Embedding(num_class+1, embed_size) 
        self.num_class = num_class
        self.start_tok = num_class  # the start token is the last id
        
        #make RNN decoder
        if cell_type.lower() == 'lstm':  self.rnn_cell = nn.LSTM
        elif cell_type.lower() == 'gru': self.rnn_cell = nn.GRU
        else: raise ValueError("unsupported RNN type")
        
        rnn_input_size = embed_size+h_size #concatentation to effectively have 2 inputs (Wy + Wh)
        self.rnn = self.rnn_cell(rnn_input_size, rnn_h_size, bidirectional=False, dropout=dropout, batch_first=True)
        self.h_size = rnn_h_size

        #output classifier
        self.classifier = nn.Linear(rnn_h_size, num_class)

    def forward(self, encoder_outputs, encoder_mask, labels):
        """teacher forcing training"""
        labels = torch.roll(labels, 1, -1)    #roll labels to use previous
        labels[:, 0] = self.start_tok         #set start token
        labels[labels==-100] = self.start_tok #pad all labels with -100
        label_embed = self.embedding(labels)  # [B, N]->[B, N, D_e]
        
        rnn_inputs  = torch.cat((encoder_outputs, label_embed), dim=-1)
        output, (hn, cn) = self.rnn(rnn_inputs)
        y = self.classifier(output)         # [B, N, D]->[B, N, 43]
        return y

    def step(self, embed, label, hx=None, cx=None):
        """steps a single input through a RNN cell. label.shape = [1]
        embed.shape = cx.shape = hx.shape = [100]"""
        
        embed       = embed.view(1,1,-1)
        label_embed = self.embedding(label).view(1,1,-1)
        rnn_input   = torch.cat((embed, label_embed), dim=-1)  # [1,1,E]
        output, (hx, cx) = self.rnn(rnn_input, (hx.view(1,1,-1), cx.view(1,1,-1)))
        y = self.classifier(output)
        return y.view(-1), hx.view(-1), cx.view(-1)
