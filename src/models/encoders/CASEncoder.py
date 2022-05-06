import torch
from torch.autograd import Variable
import torch.nn as nn
import os

from ...utils.glove_utils import get_glove

# code for Structured Self Attentive Sentence Embedding (SAS) encoder
# code based on the code found from below:
# https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding/blob/master

# a MAJOR CHANGE are that config is hardcoded
config = {
            'dropout': 0.5,        # dropout rate
            'ntoken': 141934,      # vocab size
            'nlayers': 2,          # num_layers
            'nhid': 300,           # hidden vector size
            'ninp': 300,           # embedding size
            'attention-unit': 350, # num attention units
            'attention-hops': 1,   # num attention hops
            'nfc': 512,            # num attention hops
         }

class BiLSTM(nn.Module):
    ### Changed Line #################################################################
    #def __init__(self, config):
    ##################################################################################
    def __init__(self, *args): 
        super(BiLSTM, self).__init__()
        
        self.drop = nn.Dropout(config['dropout'])
        self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], 
                              dropout=config['dropout'], bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']

        ### Put Glove loading in util function (and deleted lines) ###################
        glove_embed = get_glove()[1]
        self.embedding = nn.Embedding.from_pretrained(glove_embed)
        ##############################################################################

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]    
        ### Lines were deleted here (no need for other pooling methods) ###
        outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

class SASEncoder(nn.Module):
    def __init__(self, *args):
        super(SASEncoder, self).__init__()
        self.bilstm = BiLSTM()
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.attention_hops = config['attention-hops']

    def forward(self, trans_args):
        input_ids = trans_args['input_ids']
        print(input_ids.shape)
        outp = self.bilstm.forward(input_ids)[0]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])               # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()    # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)            # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == 0).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        #return torch.bmm(alphas, outp), alphas
        return torch.bmm(alphas, outp)
