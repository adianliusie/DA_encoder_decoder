import torch

from .hugging_utils import get_transformer
from .embed_patch.extra_embed import extra_embed


from .batchers import FlatBatcher, HierBatcher
from ..helpers.dir_manager import DirManager

#different encodersdecoders source code
from .encoders.encoders import TransUttEncoder, TransWordEncoder, HierEncoder
from .decoders import RNNDecoder, LinearHead, CRFDecoder, TransformerDecoder
from .encoders.sas_encoder import SASEncoder

class SystemHandler:
    ### methods for making batcher #####################################
    @classmethod
    def batcher(cls, encoder:str, formatting=None, max_len:int=None):
        batchers = {'utt_trans':  FlatBatcher,
                    'word_trans': FlatBatcher,
                    'hier':       HierBatcher, 
                    'sas':        HierBatcher}

        batcher = batchers[encoder](formatting=formatting,
                                    max_len=max_len)
        return batcher
    
    ### methods for making model  ###########################################
    @classmethod
    def make_seq2seq(cls, system:str, encoder:str, decoder:str,  
                     num_labels:int=None, system_args=None, C=None, **kwargs):
        """ creates the sequential classification model """

        if system == 'glove': 
            trans_model = None
        else:
            trans_model = cls.make_transformer(system, system_args, C)
        
        encoders = {'utt_trans':  TransUttEncoder, 
                    'word_trans': TransWordEncoder,
                    'hier':       HierEncoder,
                    'sas':        SASEncoder}
        
        decoders = {'linear':      LinearHead,
                    'transformer': TransformerDecoder,
                    'rnn':         RNNDecoder, 
                    'crf':         CRFDecoder}

        #select the chosen encoder and decoder
        encoder = encoders[encoder](trans_model)
        decoder = decoders[decoder](num_labels)
        model = Seq2SeqModel(encoder, decoder)
        return model
    
    @classmethod
    def make_transformer(cls, trans_name:str, system_args=None, C=None):
        """ prepares the chosen transformer """
        trans_model = get_transformer(trans_name)
        
        #add extra tokens if added into tokenizer
        if C and len(C.tokenizer) != trans_model.config.vocab_size:
            print('extending embeddings of model')
            trans_model.resize_token_embeddings(len(C.tokenizer)) 
            
        if system_args:
            trans_model = cls.patch_transformer(trans_model, trans_name, system_args)
        
        return trans_model
    
    @classmethod
    def patch_transformer(cls, trans_model, trans_name, system_args):
        options = len(system_args)
        # --system_args 8_layers ->   then only have 8 layers in transformer
        if any(['layers' in i for i in system_args]):
            layer_arg = [i for i in system_args if 'layers' in i][0]
            num_layers = int(layer_arg.split('_')[0])
            trans_model.encoder.layer = trans_model.encoder.layer[:num_layers]
            print(f'using {num_layers} transformer layers')
            options -= 1
            
        # --system_args 3_rand ->     then randomize last 3 layers of transformer
        if any(['rand' in i for i in system_args]):
            rand_arg = [i for i in system_args if 'rand' in i][0]
            num_rand_layers = int(rand_arg.split('_')[0])
            for layer in trans_model.encoder.layer[::-1][:num_rand_layers]:
                layer.apply(trans_model._init_weights)
            print(f're-initialised the last {num_rand_layers} transformer layers')
            options -= 1

        if ('spkr_embed' in system_args) or ('utt_embed' in system_args): 
            print('using speaker embeddings')
            trans_model = extra_embed(trans_model, trans_name)
            options -= 1

        assert(options == 0), "invalid system args given"
        return trans_model

    ### util methods for models ###########################################

    @classmethod
    def load_encoder(cls, model, model_path, freeze=False):
        ptrain_dir  = DirManager.load_dir(model_path)
        p_args = ptrain_dir.load_args('model_args')
        p_args = p_args.__dict__ #converts simplenamespace to dict
        ptrain_model = cls.make_seq2seq(**p_args)
        
        ptrain_model.load_state_dict(
            torch.load(ptrain_dir.path + f'/models/base.pt')
        )
        model.encoder = ptrain_model.encoder
        
        if freeze:
            for param in model.encoder.parameters():
                param.requires_grad = False
     
        freeze_str = '(frozen)' if freeze else ''
        print(f'loaded pretrained encoder {freeze_str}')
        return model
    

class Seq2SeqModel(torch.nn.Module):
    """model for dealing with Autoregressive Systems"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, trans_args, labels=None, **kwargs):
        enc_out, enc_mask = self.encoder(trans_args=trans_args, 
                                         **kwargs)
        
        y = self.decoder(encoder_outputs=enc_out,
                         encoder_mask=enc_mask, 
                         labels=labels)
        return y
   
    def decode(self, trans_args, utt_pos=None):
        enc_out, enc_mask = self.encoder(trans_args=trans_args, utt_pos=utt_pos)
        pred_labels = self.decoder.decode(encoder_outputs=enc_out, 
                                          encoder_mask=enc_mask)
        return pred_labels
    