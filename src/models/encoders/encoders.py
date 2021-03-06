import torch
import torch.nn as nn
import copy
from torch.nn.utils.rnn import pad_sequence

class TransUttEncoder(torch.nn.Module):
    """encodes all utterances jointly in a single transformer""" 
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        ### TEMP ##############################################################################################
        #self.layer = -1
        #######################################################################################################
        
    def forward(self, trans_args, utt_pos,  **kwargs):
        ### TEMP ##############################################################################################
        #H = self.transformer(output_hidden_states=True, **trans_args).hidden_states
        #H = H[self.layer] 
        #######################################################################################################
        H = self.transformer(**trans_args).last_hidden_state    #[bsz, L, 768] 
        H = self.get_sent_vectors(H, utt_pos)                   #[bsz, N, 768] 
        return H
    
    def get_sent_vectors(self, H:torch.Tensor, utt_pos_seq:'List[list]'):
        "only selects vectors at positions utt_pos_seq"
        output = [None for _ in range(len(utt_pos_seq))]
        for conv_num, utt_pos in enumerate(utt_pos_seq):
            h = H[conv_num]                         #[L, 768]
            utt_pos = utt_pos.unsqueeze(-1)         #[N, 1]
            utt_pos = utt_pos.repeat(1, H.size(-1)) #[N,768]
            conv_vecs = h.gather(0, utt_pos)        #[N,768]
            output[conv_num] = conv_vecs
            
        #pad array
        utt_embeds = pad_sequence(output, batch_first=True, padding_value=0.0)
        utt_mask = torch.all((utt_embeds!=0), dim=-1)
        return utt_embeds, utt_mask

class TransWordEncoder(torch.nn.Module):
    """encodes all words in a conversation jointly in a transformer""" 
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        
    def forward(self, trans_args, **kwargs):
        #H = self.transformer(**trans_args).last_hidden_state    #[bsz, L, 768] 
        H = self.transformer(**trans_args).hidden_states[-4]
        mask = torch.all((H!=0), dim=-1)
        return H, mask
    
class HierEncoder(torch.nn.Module):
    """wrapper where a classification head is added to trans"""
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, trans_args, conv_splits, **kwargs):
        H = self.transformer(**trans_args).last_hidden_state #[N*bsz, L, 768]
        utt_embeds = H[:, 0]                                 #[N*bsz, 768]
        
        # all utterances from all cvonersations where processed in the same tensor, 
        # now grouping the utterances back into conversations
        utt_embeds = [utt_embeds[i:j] for (i, j) in conv_splits]
        utt_embeds = pad_sequence(utt_embeds, batch_first=True, padding_value=0.0)
        utt_mask = torch.all((utt_embeds!=0), dim=-1)
        return utt_embeds, utt_mask
