import torch

from .hugging_utils import get_transformer
from .embed_patch.extra_embed import extra_embed

from .models_src.encoders import TransUttEncoder, TransWordEncoder
from .models_src.transformer_decoder import TransformerDecoder
from .models_src.rnn_decoder import DecoderRNN
from .models_src.linear_head import LinearHead

from .batchers import FlatBatcher
from ..config import config 

class SystemHandler:
    @classmethod
    def batcher(cls, system:str, formatting=None, max_len:int=None):
        batchers = {'utt_trans':  FlatBatcher,
                    'word_trans': FlatBatcher,
                    'hier':       'HierBatcher'}

        batcher = batchers[system](formatting=formatting,
                                    max_len=max_len)
        return batcher
    
    @classmethod
    def make_seq2seq(cls, transformer:str, encoder:str, decoder:str, 
                      num_labels:int=None, system_args=None, C:'ConvHandler'=None):
        """ creates the sequential classification model """

        trans_name = transformer
        trans_model = get_transformer(trans_name)

        #add extra tokens if added into tokenizer
        if len(C.tokenizer) != trans_model.config.vocab_size:
            print('extending model')
            trans_model.resize_token_embeddings(len(C.tokenizer)) 
            
        if system_args:
            trans_model = cls.patch(trans_model, trans_name, system_args)
        
        encoders = {'utt_trans':  TransUttEncoder, 
                    'word_trans': TransWordEncoder,
                    'hier':       'HierModel'}
        
        decoders = {'linear':      LinearHead,
                    'transformer': TransformerDecoder,
                    'rnn':         DecoderRNN}

        #select the chosen encoder and decoder
        encoder = encoders[encoder](trans_model)
        decoder = decoders[decoder](num_labels)
        model = Seq2SeqModel(encoder, decoder)
        return model
    
    @classmethod
    def patch(cls, trans_model, trans_name, system_args):
        if ('spkr_embed' in system_args) or ('utt_embed' in system_args): 
            print('using speaker embeddings')
            trans_model = extra_embed(trans_model, trans_name)

        if 'freeze-trans' in system_args:
            self.freeze_trans(transformer)
        
        return trans_model
             
    @staticmethod
    def freeze_trans(transformer):
        for param in transformer.encoder.parameters():
            param.requires_grad = False

class Seq2SeqModel(torch.nn.Module):
    """model for dealing with Autoregressive Systems"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, trans_args, utt_pos=None, labels=None):
        enc_out, enc_mask = self.encoder(trans_args=trans_args, 
                                            utt_pos=utt_pos)
        
        y = self.decoder(encoder_outputs=enc_out,
                         encoder_mask=enc_mask, 
                         labels=labels)
        return y
   
    def evaluate(self, trans_args, utt_pos=None):
        encoder_outputs = self.encoder(trans_args=trans_args, utt_pos=utt_pos)
        pass
    