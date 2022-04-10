from transformers.models.bart.modeling_bart import BartDecoder, shift_tokens_right
from transformers import BartConfig
import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, num_class, h_size=768, layers=2, heads=12):
        config = {'vocab_size':num_class+3, 
                  'd_model':768, 
                  'decoder_layers':layers,
                  'decoder_attention_heads':heads,
                  'decoder_start_token_id':num_class,
                  'forced_eos_token_id':num_class+1,
                  'pad_token_id':num_class+2}
        config = BartConfig(**config)

        #save special attributes
        self.start_tok = num_class
        self.end_tok = num_class+1
        self.pad_tok = num_class+2
        
        #model
        super().__init__()
        self.decoder = BartDecoder(config)
        self.classifier = nn.Linear(h_size, num_class)

    def forward(self, encoder_outputs, encoder_mask, labels):
        #roll labels for teacher forcing
        labels = torch.roll(labels, 1, -1)    
        labels[:, 0] = self.start_tok     

        #feed throug decoder
        H_dec = self.decoder.forward(
                input_ids=labels,
                attention_mask=None,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_mask,
                return_dict=True)
        H_dec = H_dec.last_hidden_state
        
        #linear classifier
        y = self.classifier(H_dec)
        return y