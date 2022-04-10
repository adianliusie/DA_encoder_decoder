import torch
import torch.nn as nn

from transformers.models.longformer.modeling_longformer import *


class LongformerEmbeddingsExtra(LongformerEmbeddings):

    # >>> added method  ########################################################################################
    def init(self):
        embed_dim = self.word_embeddings.embedding_dim
        self.speaker_embeddings = nn.Embedding(2, embed_dim)
        self.utterance_embeddings = nn.Embedding(500, embed_dim)
        
        #initiaise weights to be small, (break symmetry and starts model at working)
        self.speaker_embeddings.weight.data.uniform_(-0.001, 0.001)
        self.utterance_embeddings.weight.data.uniform_(-0.001, 0.001)

    # <<< end of method ########################################################################################

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None,       
        # >>> added arguments  ####################################################################################
        speaker_ids=None, utterance_ids=None):
        # <<< end of arguments ####################################################################################
               
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # >>> added lines  ########################################################################################
        
        if speaker_ids is None:
            speaker_embeds = torch.zeros(inputs_embeds.shape, dtype=torch.long, device=self.position_ids.device)
        else:
            speaker_embeds = self.speaker_embeddings(speaker_ids)
            
        if utterance_ids is None:
            utterance_embeds = torch.zeros(inputs_embeds.shape, dtype=torch.long, device=self.position_ids.device)
        else:
            utterance_embeds = self.utterance_embeddings(utterance_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings + speaker_embeds + utterance_embeds 
        #embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # >>> end of lines #######################################################################################

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class LongformerModelExtraEmbeds(LongformerModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        global_attention_mask=None,
        head_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        # >>> added arguments  ##################################################################################
        speaker_ids=None, 
        utterance_ids=None,
        # <<< end of arguments ##################################################################################
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)

        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pad_token_id=self.config.pad_token_id,
        )

        #>>> added lines of code  #############################################################################
        if utterance_ids is not None:
            utterance_ids = nn.functional.pad(utterance_ids, (0, padding_len), value=0)
        #<<< end of lines    ##################################################################################  
        
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[
            :, 0, 0, :
        ]

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            #>>> added arguments  #############################################################################
            speaker_ids=speaker_ids, utterance_ids=utterance_ids
            #<<< end of arguments #############################################################################
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # undo padding
        if padding_len > 0:
            # unpad `sequence_output` because the calling function is expecting a length == input_ids.size(1)
            sequence_output = sequence_output[:, :-padding_len]

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return LongformerBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            global_attentions=encoder_outputs.global_attentions,
        )
