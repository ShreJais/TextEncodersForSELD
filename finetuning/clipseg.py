import torch
import torch.nn as nn

import transformers
import numpy as np

class CLIPSegmentation(transformers.CLIPSegForImageSegmentation):
    # It inherits all functionality of CLIPSegForImageSegmentation (e.g., forward pass, parameters, layers).
    # The constructor (__init__) just passes all arguments (*args, **kwargs) directly to the parent class 
    # via super().__init__().
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _build_causal_attention_mask(batch_size, seq_len, dtype):
        mask= torch.full(size= (batch_size, seq_len, seq_len), fill_value= float('-inf'), dtype= dtype)
        mask= torch.triu(input= mask, diagonal= 1) # Mask out future positions
        return mask
    
    def resize_token_embeddings(self, new_vocab_size):
        old_embedding= self.clip.text_model.embeddings.token_embedding
        old_num_tokens, embedding_dim= old_embedding.weight.shape

        if new_vocab_size <= old_num_tokens:
            raise ValueError(f'New vocab size: {new_vocab_size} must be > old_vocab_size: {old_num_tokens}')
        
        # create a new embedding layer.
        new_embedding= nn.Embedding(num_embeddings= new_vocab_size, embedding_dim= embedding_dim)
        # Copy over old weights.
        new_embedding.weight.data[:old_num_tokens]= old_embedding.weight.data
        # init new token rows.
        nn.init.normal_(new_embedding.weight.data[old_num_tokens:], mean= 0, std= 0.02)

        # replace in model
        self.clip.text_model.embeddings.token_embedding= new_embedding

    def decoder_forward(self, hidden_states: tuple[torch.Tensor], conditional_embeddings: torch.Tensor, resolution= (176, 352)):
        activations= hidden_states[::-1]
        output= None
        for i, (activation, layer, reduce) in enumerate(zip(activations, self.decoder.layers, self.decoder.reduces)):
            if output is not None:
                output= reduce(activation) + output
            else:
                output= reduce(activation)
            
            if i==self.decoder.conditional_layer:
                output= (self.decoder.film_mul(conditional_embeddings)*output.permute(1, 0, 2) + 
                            self.decoder.film_add(conditional_embeddings))
                output= output.permute(1, 0, 2)
            
            layer_outputs= layer(output, attention_mask= None, causal_attention_mask= None)
            output= layer_outputs[0]

        # Remove the CLS token and reshape using actual patch grid.
        output= output[:, 1:, :].permute(0, 2, 1)
        h, w= resolution[0]//self.clip.vision_model.config.patch_size, resolution[1]//self.clip.vision_model.config.patch_size
        batch_size= output.shape[0]

        output= output.view(batch_size, -1, h, w)
        logits= self.decoder.transposed_convolution(output)
        return logits

    def get_pixels(self, img):
        """
        Extract spatial features (pixel-level) from the CLIP image encoder.
        """
        vision_outputs= self.clip.vision_model(pixel_values= img, output_attentions= None, 
            output_hidden_states= True, interpolate_pos_encoding= True, return_dict= True)
        last_layer= self.clip.vision_model.encoder.layers[-1]
        hidden_states= vision_outputs.hidden_states[-2]
        residual= hidden_states
        batch_size, target_len, embed_dim= hidden_states.size()

        # get query proj
        # query_states = last_layer.self_attn.q_proj(hidden_states) * last_layer.self_attn.scale
        # key_states = last_layer.self_attn.k_proj(hidden_states)
        value_states= last_layer.self_attn.v_proj(hidden_states)
        value_states= last_layer.self_attn.out_proj(value_states)

        value_states+= residual

        residual= value_states
        value_states= last_layer.layer_norm2(value_states)
        value_states= last_layer.mlp(value_states)
        value_states+= residual

        value_states= self.clip.vision_model.post_layernorm(value_states)
        output= self.clip.visual_projection(value_states)

        # width= int(np.sqrt(target_len-1))
        width= img.shape[-1]//self.clip.vision_model.embeddings.patch_embedding.kernel_size[0]
        height= img.shape[-2]//self.clip.vision_model.embeddings.patch_embedding.kernel_size[0]
        
        output= output[:, 1:]
        if output.ndim==2:
            output= output.squeeze(0)
        
        output= output.permute(0, 2, 1)
        output= output.reshape(batch_size, self.clip.visual_projection.out_features, height, width)
        return output

    # def encode_audio(self, placeholder_token, audio_token, pos: int, length:int):
    #     """
    #     Encode audio token into the audio-driven embeddings. (Audio-Driven Embedder)

    #     Args:
    #         placeholder_token (torch.Tensor): Placeholder text token tensor.
    #         audio_token (torch.Tensor): Audio token tensor.
    #         pos (int): Position index for audio token.
    #         length (int): Length of the input token.

    #     Returns:
    #         torch.Tensor: Audio-driven embeddings.
    #     """
    #     tokens= placeholder_token

    #     if placeholder_token.ndim==3:
    #         tokens= torch.squeeze(placeholder_token, dim=1)
        
    #     inputs_embed= self.clip.text_model.embeddings.token_embedding(tokens).type(self.dtype) # (batch_size, ctx, d_model)
    #     inputs_embed= torch.cat((inputs_embed[:, :pos, :], audio_token, inputs_embed[:, :pos:, :]), dim=1)
    #     inputs_embed= inputs_embed[:, :length, :]

    #     batch_size, seq_len, _= inputs_embed.shape
    #     attention_mask= torch.ones((batch_size, seq_len)).to(placeholder_token.device)
    #     position_ids= torch.arange(length).unsqueeze(0).to(placeholder_token.device)

    #     position_embeddings= self.clip.text_model.embeddings.position_embedding(position_ids)

    #     hidden_states= inputs_embed + position_embeddings   
    #     causal_attention_mask= self.clip.text_model._build_causal_attention_mask(batch_size, seq_len, hidden_states.dtype).to(hidden_states.device)

    #     if attention_mask is not None:
    #         # (batch_size, seq_len) --> (batch_size, 1, target_seq_len, source_seq_len)
    #         attention_mask= _prepare_4d_attention_mask(mask= attention_mask, dtype= hidden_states.dtype)
    #     encoder_outputs= self.clip.text_model.encoder(
    #         inputs_embed= hidden_states, attention_mask= attention_mask,
    #         causal_attention_mask= causal_attention_mask, output_attention= False, 
    #         output_hidden_states= False, return_dict= True)
        
    #     last_hidden_states= encoder_outputs[0]
    #     last_hidden_states= self.clip.text_model.final_layer_norm(last_hidden_states)

    #     pooled_output= last_hidden_states[:, -1, :]
    #     audio_driven_embeddings= self.clip.text_projection(pooled_output)
    #     return audio_driven_embeddings