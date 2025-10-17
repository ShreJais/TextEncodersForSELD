import torch
import torch.nn as nn
import torch.nn.functional as f

import numpy as np

from typing import Optional, Dict, Tuple
import math, warnings
warnings.filterwarnings("ignore")

from torchinfo import summary

class BEATSConfig:
    def __init__(self, cfg):
        self.beats_cfg= cfg

class BEATS(nn.Module):
    def __init__(self, cfg: BEATSConfig):
        super().__init__()
        self.beats_cfg= cfg.beats_cfg
        self.post_extract_proj= (nn.Linear(in_features= self.beats_cfg['embed_dim'], 
            out_features= self.beats_cfg['encoder_embed_dim']) if self.beats_cfg['embed_dim'] != self.beats_cfg['encoder_embed_dim'] else None)
        self.patch_embedding= nn.Conv2d(in_channels= 1, out_channels= self.beats_cfg['embed_dim'], 
            kernel_size= self.beats_cfg['input_patch_size'], stride= self.beats_cfg['input_patch_size'], bias= self.beats_cfg['conv_bias'])
        self.input_dropout= nn.Dropout(p = self.beats_cfg['input_dropout_rate'])
        
        assert not self.beats_cfg['deep_norm'] or not self.beats_cfg['layer_norm_first']
        self.encoder= TransformerEncoder(cfg= self.beats_cfg)
        self.layer_norm= nn.LayerNorm(normalized_shape= self.beats_cfg['embed_dim'])

        if self.beats_cfg['finetuned_model']:
            self.predictor_dropout= nn.Dropout(p= self.beats_cfg['predictor_dropout_rate'])
            self.predictor= nn.Linear(in_features= self.beats_cfg['encoder_embed_dim'], out_features= self.beats_cfg['predictor_class'])
        else:
            self.predictor= None

    def forward_padding_mask(self, features: torch.Tensor, padding_mask: torch.Tensor):
        extra= padding_mask.size(1)%features.size(1)
        if extra > 0:
            padding_mask= padding_mask[:, :-extra]
        padding_mask= padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask= padding_mask.all(-1)
        return padding_mask
    
    def extract_features(self, source: torch.Tensor, normalize= False, padding_mask: Optional[torch.Tensor]= None, 
        fbank_mean: float= 15.41663, fbank_std: float= 6.55582):
        # fbank= self.preprocess(source= source, fbank_mean= fbank_mean, fbank_std= fbank_std)
        # if padding_mask is not None:
        #     padding_mask= self.forward_padding_mask(features= fbank, padding_mask= padding_mask)
        # source= ()
        # fbank= fbank.unsqueeze(1)
        if normalize:
            source= (source - fbank_mean)/(2*fbank_std)
        features= self.patch_embedding(source)
        features= features.transpose(1, 3)
        features= features.reshape(-1, features.shape[2], features.shape[3])
        # features= features.reshape(features.shape[0], features.shape[1], -1)
        # features= features.transpose(1, 2)
        features= self.layer_norm(features)
        
        if padding_mask is not None:
            padding_mask= self.forward_padding_mask(features= features, padding_mask= padding_mask)
        
        if self.post_extract_proj is not None:
            features= self.post_extract_proj(features)

        x= self.input_dropout(features)

        x, layers_cat, layers= self.encoder(x, padding_mask= padding_mask, layer= None)
        if self.predictor is not None:
            x= self.predictor_dropout(x)
            logits= self.predictor(x)

            if padding_mask is not None and padding_mask.any():
                logits[padding_mask]= 0
                logits= logits.sum(dim=1)
                logits= logits/(-padding_mask).sum(dim=1).unsqueeze(-1).expand_as(logits)
            else:
                logits= logits.mean(dim= 1)
            lprobs= torch.sigmoid(logits)
            return lprobs, padding_mask
        else:
            return x, layers_cat, layers

class TransformerEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg= cfg
        self.pos_conv= nn.Conv1d(in_channels= self.cfg['encoder_embed_dim'], 
            out_channels= self.cfg['encoder_embed_dim'], kernel_size= self.cfg['conv_pos'], 
            padding= self.cfg['conv_pos']//2, groups= self.cfg['conv_pos_groups'])
        
        dropout_rate= 0.
        std= math.sqrt((4*(1-dropout_rate))/(self.cfg['conv_pos'] * self.cfg['encoder_embed_dim']))
        nn.init.normal_(self.pos_conv.weight, mean= 0, std= std)
        nn.init.constant_(self.pos_conv.bias, val=0)

        self.pos_conv= nn.utils.parametrizations.weight_norm(
            module= self.pos_conv, name= 'weight', dim= 2)
        self.pos_conv= nn.Sequential(
                            self.pos_conv, 
                            SamePad(kernel_size= self.cfg['conv_pos']),
                            nn.GELU())
        
        if 'relative_positional_embedding' in self.cfg:
            self.relative_positional_embedding= self.cfg['relative_positional_embedding']
            self.num_buckets= self.cfg['num_buckets']
            self.max_dist= self.cfg['max_dist']
        else:
            self.relative_positional_embedding= False
            self.num_buckets= 0
            self.max_dist= 0
        
        self.layers= nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim= self.cfg['encoder_embed_dim'], ffn_embedding_dim= self.cfg['encoder_ffn_embed_dim'],
                num_attention_heads= self.cfg['encoder_attention_heads'], dropout_rate= self.cfg['dropout_rate'], 
                attention_dropout_rate= self.cfg['attention_dropout_rate'], 
                activation_dropout_rate= self.cfg['actfn_dropout_rate'], activation_fn= self.cfg['act_fn'],
                layer_norm_first= self.cfg['layer_norm_first'], deep_norm= self.cfg['deep_norm'],
                has_relative_attention_bias= self.cfg['relative_positional_embedding'], 
                num_buckets= self.cfg['num_buckets'], max_distance= self.cfg['max_dist'], 
                rescale_init= self.cfg['rescale_init'], gru_rel_pos= self.cfg['gru_rel_pos'], 
                encoder_layers= self.cfg['encoder_layers']) 
            for i in range(self.cfg['encoder_layers'])])
        
        # if self.relative_positional_embedding:
        #     for i in range(1, self.cfg['encoder_layers']):
        #         del self.layers[i].self_attn.relative_attention_bias
        #         self.layers[i].self_attn.relative_attention_bias= self.layers[0].self_attn.relative_attention_bias

        self.layer_norm_first= self.cfg['layer_norm_first']
        self.layer_norm= nn.LayerNorm(normalized_shape= self.cfg['encoder_embed_dim'])
        self.encoderlayer_dropout_rate= self.cfg['encoderlayer_dropout_rate']

        self.apply(init_bert_params)

        if self.cfg['deep_norm']:
            deepnorm_beta= math.pow(8*self.cfg['encoder_layers'], -1/4)
            for i in range(self.cfg['encoder_layers']):
                nn.init.xavier_normal_(self.layers[i].self_attn.k_proj.weight, gain= 1)
                nn.init.xavier_normal_(self.layers[i].self_attn.v_proj.weight, gain= deepnorm_beta)
                nn.init.xavier_normal_(self.layers[i].self_attn.q_proj.weight, gain= 1)
                nn.init.xavier_normal_(self.layers[i].self_attn.out_proj.weight, gain= deepnorm_beta)
                nn.init.xavier_normal_(self.layers[i].fc1.weight, gain= deepnorm_beta)
                nn.init.xavier_normal_(self.layers[i].fc2.weight, gain= deepnorm_beta)
                
        self.layer_wise_gradient_decay_ratio= self.cfg['layer_wise_gradient_decay_ratio']

    def forward(self, x, padding_mask= None, layer= None):
        x, layers_cat, layers= self.extract_features(x=x, padding_mask= padding_mask, target_layer= layer)
        if self.layer_norm_first and layer is None:
            x= self.layer_norm(x)
        return x, layers_cat, layers
    
    def extract_features(self, x, padding_mask= None, target_layer= None):
        if padding_mask is not None:
            x[padding_mask]= 0
        
        x_conv= self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x= x+ x_conv

        if not self.layer_norm_first:
            x= self.layer_norm(x)
        x= f.dropout(x, p= self.cfg['dropout_rate'], training= self.training)

        # batch_size, seq_len, ch --> seq_len, batch_size, ch
        x= x.transpose(0, 1)
        layers, layer_results, z = [], [], None

        if target_layer is not None:
            layer_results.append((x, z))
        r, pos_bias= None, None

        for i, layer in enumerate(self.layers):
            if self.layer_wise_gradient_decay_ratio!= 1.:
                x= GradMultiply.apply(x, self.layer_wise_gradient_decay_ratio)
            dropout_prob= np.random.random()

            if not self.training or (dropout_prob > self.encoderlayer_dropout_rate):
                x, z, pos_bias= layer(x=x, self_attention_mask= None, 
                    self_attention_padding_mask= padding_mask, need_weights= False, pos_bias= pos_bias)
            if target_layer is not None:
                layer_results.append((x, z))
            if i==target_layer:
                r= x
                break
            if i in [3, 7, 11]:
                layers.append(x.transpose(0, 1))
        if r is not None:
            x= r
        
        # TxBxC -> BxTxC
        x= x.transpose(0, 1)
        layers_cat= torch.cat(layers, dim=2)
        return x, layers_cat, layers
    
class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, embedding_dim:int= 768, ffn_embedding_dim:int= 3072, num_attention_heads:int= 12, 
        dropout_rate:float= 0.0, attention_dropout_rate:float= 0.0, activation_dropout_rate:float= 0.0,
        activation_fn:str= 'gelu', layer_norm_first:bool= False, deep_norm:bool= True, 
        has_relative_attention_bias:bool= True, num_buckets:int= 320, max_distance:int= 800, 
        rescale_init:bool= False, gru_rel_pos:bool= True, encoder_layers:int= 12):
        
        super().__init__()
        self.embedding_dim= embedding_dim
        self.dropout_rate= dropout_rate
        self.activation_dropout_rate= activation_dropout_rate

        self.act_fn_name= activation_fn
        self.act_fn= get_activation_fn(act_fn_name= self.act_fn_name)
        self.self_attn= MultiheadAttention(embed_dim= self.embedding_dim, num_heads= num_attention_heads, 
            kdim= None, vdim= None, dropout_rate= self.dropout_rate, bias= True, add_bias_kv= False, 
            add_zero_attn= False, self_attention= True, encoder_decoder_attention= False, q_noise= 0, 
            qn_block_size= 8, has_relative_attention_bias= has_relative_attention_bias, num_buckets= num_buckets,
            max_distance= max_distance, rescale_init= rescale_init, gru_rel_pos= gru_rel_pos)
        
        self.dropout1= nn.Dropout(p = self.dropout_rate)
        self.dropout2= nn.Dropout(p= self.activation_dropout_rate)
        self.dropout3= nn.Dropout(p = self.dropout_rate)

        self.layer_norm_first= layer_norm_first
        self.self_attn_layer_norm= nn.LayerNorm(self.embedding_dim)

        if self.act_fn_name=='glu':
            self.fc1= GLU_linear(input_dim= self.embedding_dim, output_dim= ffn_embedding_dim, glu_type= 'swish')
        else:
            self.fc1= nn.Linear(in_features= self.embedding_dim, out_features= ffn_embedding_dim)
        self.fc2= nn.Linear(in_features= ffn_embedding_dim, out_features= self.embedding_dim)

        self.final_layer_norm= nn.LayerNorm(normalized_shape= self.embedding_dim)

        self.deep_norm= deep_norm
        if self.deep_norm:
            self.deepnorm_alpha= math.pow(2*encoder_layers, 1/4)
        else:
            self.deepnorm_alpha= 1
        
    def forward(self, x: torch.Tensor, self_attention_mask: torch.Tensor= None, 
        self_attention_padding_mask: torch.Tensor= None, need_weights: bool= True, pos_bias= None):
        residual= x
        if self.layer_norm_first:
            x= self.self_attn_layer_norm(x)
            x, attn, pos_bias= self.self_attn(query= x, key= x, value=x, 
                key_padding_mask= self_attention_padding_mask, incremental_state= None, 
                need_weights= need_weights, static_kv= False, attn_mask= self_attention_mask, 
                before_softmax= False, need_head_weights= False, position_bias= pos_bias)
            x= self.dropout1(x)
            x= residual+x

            residual= x
            x= self.final_layer_norm(x)
            if self.act_fn_name=='glu':
                x= self.fc1(x)
            else:
                x= self.act_fn(self.fc1(x))
            x= self.dropout3(self.fc2(self.dropout2(x)))
            x= residual+x

        else:
            x, attn, pos_bias= self.self_attn(query= x, key= x, value=x, 
                key_padding_mask= self_attention_padding_mask, incremental_state= None, 
                need_weights= need_weights, static_kv= False, attn_mask= self_attention_mask, 
                before_softmax= False, need_head_weights= False, position_bias= pos_bias)
            x= self.dropout1(x)
            x= residual*self.deepnorm_alpha+x
            x= self.self_attn_layer_norm(x)
            
            residual= x
            if self.act_fn_name=='glu':
                x= self.fc1(x)
            else:
                x= self.act_fn(self.fc1(x))
            x= self.dropout3(self.fc2(self.dropout2(x)))
            x= residual*self.deepnorm_alpha+x
            x= self.final_layer_norm(x)
        return x, attn, pos_bias

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim= 768, num_heads= 12, kdim= None, vdim= None, dropout_rate= 0.0, 
        bias= True, add_bias_kv= False, add_zero_attn= False, self_attention= True, 
        encoder_decoder_attention= False, q_noise= 0.0, qn_block_size= 8, 
        has_relative_attention_bias= True, num_buckets= 320, max_distance= 800, 
        gru_rel_pos= True, rescale_init= False):

        super().__init__()
        self.embed_dim= embed_dim
        self.kdim= kdim if kdim is not None else self.embed_dim
        self.vdim= vdim if vdim is not None else self.embed_dim
        self.qkv_same_dim= (self.kdim==self.embed_dim) and (self.vdim==self.embed_dim)
        self.num_heads= num_heads
        self.dropout_module= nn.Dropout(p= dropout_rate)

        self.has_relative_attention_bias= has_relative_attention_bias
        self.num_buckets= num_buckets
        self.max_distance= max_distance

        if self.has_relative_attention_bias:
            self.relative_attention_bias= nn.Embedding(num_embeddings= self.num_buckets, embedding_dim= self.num_heads)
        self.head_dim= self.embed_dim//self.num_heads
        self.q_head_dim= self.head_dim
        self.k_head_dim= self.head_dim

        assert self.head_dim*self.num_heads == self.embed_dim, 'embed dim must be divisible by num_heads.'
        self.scaling= self.head_dim**-0.5
        self.self_attention= self_attention
        self.encoder_decoder_attention= encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, 'Self-attention requires query, key and value to be of the same size.'
        
        k_bias= True
        if rescale_init:
            k_bias= False

        k_embed_dim= embed_dim
        q_embed_dim= embed_dim

        self.k_proj= quant_noise(module= nn.Linear(in_features= self.kdim, out_features= k_embed_dim, bias= k_bias),
            p= q_noise, block_size= qn_block_size)
        self.v_proj= quant_noise(module= nn.Linear(in_features= self.vdim, out_features= embed_dim, bias= bias),
            p= q_noise, block_size= qn_block_size)
        self.q_proj= quant_noise(module= nn.Linear(in_features= embed_dim, out_features= q_embed_dim, bias= bias),
            p= q_noise, block_size= qn_block_size)
        self.out_proj= quant_noise(module= nn.Linear(in_features= embed_dim, out_features= embed_dim, bias= bias),
            p= q_noise, block_size= qn_block_size)
        
        if add_bias_kv:
            self.bias_k= nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v= nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k= self.bias_v= None
        
        self.add_zero_attn= add_zero_attn
        self.gru_rel_pos= gru_rel_pos
        if self.gru_rel_pos:
            self.grep_linear= nn.Linear(in_features= self.q_head_dim, out_features= 8)
            self.grep_a= nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)

    def forward(self, query, key: Optional[torch.Tensor], value: Optional[torch.Tensor], 
        key_padding_mask: Optional[torch.Tensor]= None, 
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]= None,
        need_weights: bool= True, static_kv: bool= False, attn_mask: Optional[torch.Tensor]= None,
        before_softmax: bool= False, need_head_weights: bool= False, 
        position_bias: Optional[torch.Tensor]= None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights= True

        is_tpu= query.device.type == 'xla'
        target_len, batch_size, embed_dim= query.size()
        source_len= target_len

        assert embed_dim==self.embed_dim
        assert list(query.size())==[target_len, batch_size, embed_dim]

        if key is not None:
            source_len, key_batch_size, _= key.size()
            if not torch.jit.is_scripting():
                assert key_batch_size==batch_size
                assert value is not None
                assert source_len, batch_size== value.shape[:2]
        
        if self.has_relative_attention_bias and position_bias is None:
            position_bias= self.compute_bias(query_len= target_len, key_len= source_len)
            position_bias= position_bias.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(batch_size*self.num_heads, target_len, source_len)

        if incremental_state is not None:
            saved_state= self._get_input_buffer(incremental_state)
            if saved_state is not None and 'prev_key' in saved_state:
                # previous time steps are cached  - no need to recompute key and value if they static.
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key= value= None
        else:
            saved_state= None

        if self.self_attention:
            q, k, v= self.q_proj(query), self.k_proj(query), self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention.
            if key is None:
                assert value is None
                k= v= None
            else:
                k, v= self.k_proj(key), self.v_proj(key)
        else:
            assert key is not None and value is not None
            q, k, v= self.q_proj(query), self.k_proj(key), self.v_proj(value)
        
        alpha= 32
        q= (1/alpha)*q*self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k= torch.cat([k, self.bias_k.repeat(1, batch_size, 1)])
            v= torch.cat([v, self.bias_v.repeat(1, batch_size, 1)])

            if attn_mask is not None:
                attn_mask= torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask= torch.cat([key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
        q= q.contiguous().view(target_len, batch_size*self.num_heads, self.q_head_dim).transpose(0, 1)

        if k is not None:
            k= k.contiguous().view(-1, batch_size*self.num_heads, self.k_head_dim).transpose(0, 1)
        if v is not None:
            v= v.contiguous().view(-1, batch_size*self.num_heads, self.k_head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved state are stored with shape: (batch_size, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key= saved_state['prev_key']
                assert _prev_key is not None
                prev_key= _prev_key.view(batch_size*self.num_heads, -1, self.head_dim)
                if static_kv:
                    k= prev_key
                else:
                    assert k is not None
                    k= torch.cat([prev_key, k], dim=1)
                source_len= k.size(1)
            if 'prev_value' in saved_state:
                _prev_value= saved_state['prev_value']
                assert _prev_value is not None
                prev_value= _prev_value.view(batch_size*self.num_heads, -1, self.head_dim)
                if static_kv:
                    v= prev_value
                else:
                    assert v is not None
                    v= torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[torch.Tensor]= None

            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask= saved_state['prev_key_padding_mask']
            assert k is not None and v is not None

            key_padding_mask= MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask= key_padding_mask, prev_key_padding_mask= prev_key_padding_mask, 
                batch_size= batch_size, source_len= source_len, static_kv= static_kv)
            
            saved_state['prev_key']= k.view(batch_size, self.num_heads, -1, self.head_dim)
            saved_state['prev_value']= v.view(batch_size, self.num_heads, -1, self.head_dim)
            saved_state['prev_key_padding_mask']= key_padding_mask


            # In this branch  incremental_state is never None.
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        
        assert k is not None
        assert k.size(1) == source_len

        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0)==batch_size
            assert key_padding_mask.size(1)==source_len
        
        if self.add_zero_attn:
            assert v is not None
            source_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask= torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask= torch.cat([key_padding_mask, 
                    torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)


        attn_weights= torch.bmm(q, k.transpose(1, 2))
        attn_weights= (attn_weights-attn_weights.max(dim=-1, keepdim=True)[0])*alpha
        attn_weights= self.apply_sparse_mask(attn_weights= attn_weights, tgt_len= target_len, src_len= source_len, bsz= batch_size)
        
        assert attn_weights.size() == (batch_size*self.num_heads, target_len, source_len)

        if attn_mask is not None:
            attn_mask= attn_mask.unsqueeze(0)
            attn_weights+= attn_mask
        
        if key_padding_mask is not None:
            # don't attend the padding symbols.
            attn_weights= attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            if not is_tpu:
                attn_weights= attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool). float('-inf'))
            else:
                attn_weights= attn_weights.transpose(0, 2)
                attn_weights= attn_weights.masked_fill(key_padding_mask, float('-inf'))
                attn_weights= attn_weights.transpose(0, 2)
            attn_weights= attn_weights.view(batch_size*self.num_heads, target_len, source_len)

        if before_softmax:
            return attn_weights, v, position_bias

        if position_bias is not None:
            attn_mask_rel_pos= position_bias
            if self.gru_rel_pos:
                query_layer= q.view(batch_size, self.num_heads, target_len, self.q_head_dim)*alpha/self.scaling
                _B, _H, _L, __ = query_layer.size()
                gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
                gate_a_1 = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
                attn_mask_rel_pos = gate_a_1.view(batch_size*self.num_heads, target_len, 1)*position_bias

            attn_mask_rel_pos= attn_mask_rel_pos.view(attn_weights.size())
            attn_weights = attn_weights + attn_mask_rel_pos

        attn_weights_float= f.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs= self.dropout_module(attn_weights)

        assert v is not None
        attn= torch.bmm(attn_probs, v)

        assert attn.size()==(batch_size*self.num_heads, target_len, self.head_dim)
        attn= attn.transpose(0, 1).contiguous().view(target_len, batch_size, embed_dim)

        attn= self.out_proj(attn)
        attn_weights: Optional[torch.Tensor]= None
        if need_weights:
            attn_weights= attn_weights_float.view(batch_size, self.num_heads, target_len, source_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads.
                attn_weights= attn_weights.mean(dim=0)
        return attn, attn_weights, position_bias
    
    def compute_bias(self, query_len, key_len):
        # It generates relative position attention bias used in the attention mechanism to give the model 
        # a sense of relative position between tokens -- how far apart two tokens are, rather than their 
        # absolute positions. This is more robust and generalizable, especially for tasks involving variable-length
        # sequence.
        context_position= torch.arange(query_len, dtype= torch.long)[:, None]
        memory_position= torch.arange(key_len, dtype= torch.long)[None, :]
        relative_position= memory_position-context_position
        relative_position_bucket= self._relative_position_bucket(relative_position= relative_position, bidirectional= True)
        relative_position_bucket= relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values= self.relative_attention_bias(relative_position_bucket)
        values= values.permute([2, 0, 1])
        return values
    
    def _relative_position_bucket(self, relative_position, bidirectional= True):
        num_buckets= self.num_buckets
        max_distance= self.max_distance
        relative_buckets= 0

        if bidirectional:
            num_buckets= num_buckets//2
            relative_buckets+= (relative_position>0).to(torch.long)*num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position= -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact= num_buckets//2
        is_small= relative_position < max_exact

        relative_position_if_large= (max_exact + 
            torch.log(relative_position.float()/max_exact)/math.log(max_distance/max_exact)*(num_buckets-max_exact)).to(torch.long)
        relative_position_if_large= torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, fill_value= (num_buckets-1)))
        relative_buckets+= torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def _get_input_buffer(self, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]):
        result= self.get_incremental_state(incremental_state, 'attn_state')
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[torch.Tensor]]= {}
            return empty_result
    
    def _set_input_buffer(self, incremental_state: Dict[str, Dict[str, Optional[torch.Tensor]]],
            buffer: Dict[str, Optional[torch.Tensor]]):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)
    
    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights
    
    @staticmethod
    def _append_prev_key_padding_mask(key_padding_mask: Optional[torch.Tensor], 
        prev_key_padding_mask: Optional[torch.Tensor], batch_size: int, 
        source_len: int, static_kv: bool)-> Optional[torch.Tensor]:
        # saved key padding masks have shape: (batch_size, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask= prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask= torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        
        # During incremental decoding, as the padding token enters and leaves the frame, 
        # there will be a time when prev or current is None.
        elif prev_key_padding_mask is not None:
            if source_len > prev_key_padding_mask.size(1):
                filter= torch.zeros((batch_size, source_len-prev_key_padding_mask.size(1)), device= prev_key_padding_mask.device)
                new_key_padding_mask= torch.cat([prev_key_padding_mask.float(), filter.float()], dim=1)
            else:
                new_key_padding_mask= prev_key_padding_mask.float()
        elif key_padding_mask is not None:
            if source_len > key_padding_mask.size(1):
                filter= torch.zeros((batch_size, source_len-key_padding_mask.size(1)), device= key_padding_mask.device)
                new_key_padding_mask= torch.cat([filter.float(), key_padding_mask.float()], dim=1)
            else:
                new_key_padding_mask= key_padding_mask.float()
        else:
            new_key_padding_mask= prev_key_padding_mask
        return new_key_padding_mask

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale= scale
        res= x.new(x)
        return res
    @staticmethod
    def backward(ctx, grad):
        return grad*ctx.scale, None

class SamePad(nn.Module):
    def __init__(self, kernel_size, causal= False):
        super().__init__()
        if causal:
            self.remove= kernel_size-1
        else:
            self.remove=1 if kernel_size%2==0 else 0
    def forward(self, x):
        if self.remove > 0:
            x=x[:, :, :-self.remove]
        return x

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.actfn= nn.Sigmoid()

    def forward(self, x):
        return x*self.actfn(x)
    
class GLU_linear(nn.Module):
    def __init__(self, input_dim, output_dim, glu_type= 'sigmoid', bias_in_glu= True):
        super().__init__()
        self.glu_type= glu_type
        self.output_dim= output_dim

        if glu_type=='sigmoid':
            self.glu_actfn= nn.Sigmoid()
        elif glu_type=='swish':
            self.glu_actfn= Swish()
        elif glu_type=='relu':
            self.glu_actfn= nn.ReLU()
        elif glu_type=='gelu':
            self.glu_actfn= nn.GELU()
        
        if bias_in_glu:
            self.linear= nn.Linear(in_features= input_dim, out_features= output_dim*2, bias= True)
        else:
            self.linear= nn.Linear(in_features= input_dim, out_features= output_dim*2, bias= True)
    def forward(self, x):
        x= self.linear(x)
        if self.glu_type=='bilinear':
            x= x[:, :, 0:self.output_dim]*x[:, :, self.output_dim:self.output_dim*2]
        else:
            x= x[:, :, 0:self.output_dim]*self.glu_actfn(x[:, :, self.output_dim:self.output_dim*2]) 
        return x

def get_activation_fn(act_fn_name: str):
    if act_fn_name=='relu':
        return f.relu
    elif act_fn_name=='gelu':
        return gelu
    elif act_fn_name=='gelu_accurate':
        return gelu_accurate
    elif act_fn_name=='tanh':
        return torch.tanh
    elif act_fn_name=='linear':
        return lambda x: x
    elif act_fn_name=='glu':
        return lambda x: x
    
def gelu(x: torch.Tensor)->torch.Tensor:
    return f.gelu(x.float()).type_as(x)

def gelu_accurate(x):
    if not hasattr(gelu_accurate, "_a"):
        gelu_accurate._a= math.sqrt(2/math.pi)
    return 0.5*x*(1+torch.tanh(gelu_accurate._a*(x+0.044715*torch.pow(x, 3))))

def quant_noise(module, p, block_size):
    """
    Wraps modules and applies quantization noise to the weights for subsequent 
    quantization with iterative product qunatization as described in 
    'Training with Quantization noise for extreme model compression'
    Args:
        - module: nn.Module
        - p: amount of Quantization Noise
        - block_size: size of the blocks for subsequent quantization with iPQ

    Remarks:
        - Module weights must have the right sizes wrt the block size
        - Only Linear, Embedding and Conv2d modules are supported for the moment
        - For more detail on how to quantize by blocks with convolutional weights,
          see "And the Bit Goes Down: Revisiting the Quantization of Neural Networks"
        - We implement the simplest form of noise here as stated in the paper
          which consists in randomly dropping blocks
    """
    # if no quantization noise, don't register hook.
    if p<=0:
        return module
    # supported modules.
    assert isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d))

    # test whether module.weight has the right sizes wrt block_size
    is_conv= module.weight.ndim == 4

    # 2D matrix
    if not is_conv:
        assert module.weight.size(dim= 1) % block_size == 0, "Input features must be a multiple of block sizes."
    # 4D matrix
    else:
        # 1x1 convolutions.
        if module.kernel_size == (1, 1):
            assert module.in_channels%block_size==0, "Input channels must be multiple of block sizes."
        # regular convolutions.
        else:
            k = module.kernel_size[0]*module.kernel_size[1]
            assert k%block_size==0, "Kernel size must be a multiple of block size"
        
    def _forward_pre_hook(mod, input):
        # no noise for evaluation.
        if mod.training:
            if not is_conv:
                # gather weight and sizes
                weight= mod.weight
                in_features= weight.size(1)
                out_features= weight.size(0)

                # split weight matrix into blocks and randomly drop selected blocks.
                mask= torch.zeros(in_features//block_size*out_features, device= weight.device)
                mask.bernoulli_(p)
                mask= mask.repeat_interleave(block_size, -1).view(-1, in_features)
            else:
                # gather weight and sizes
                weight = mod.weight
                in_channels = mod.in_channels
                out_channels = mod.out_channels

                # split weight matrix into blocks and randomly drop selected blocks
                if mod.kernel_size==(1, 1):
                    mask = torch.zeros(int(in_channels // block_size * out_channels),device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.repeat_interleave(block_size, -1).view(-1, in_channels)
                else:
                    mask = torch.zeros(weight.size(0), weight.size(1), device=weight.device)
                    mask.bernoulli_(p)
                    mask = mask.unsqueeze(2).unsqueeze(3).repeat(1, 1, mod.kernel_size[0], mod.kernel_size[1])
            # scale weihts and apply mask.
            mask= mask.to(torch.bool) #x.bool() is not currently supported in torchscript
            s= 1/(1-p)
            mod.weight.data= s*weight.masked_fill(mask, 0)
    module.register_forward_pre_hook(_forward_pre_hook)
    return module

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)

if __name__=="__main__":
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    import utils
    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    # Make changes on this yml file for different experiment.
    config_file = './configs/beats_config.yml'

    cfg= utils.get_configurations(config_file= config_file)
    
    beats_cfg= BEATSConfig(cfg= cfg)
    model= BEATS(cfg= beats_cfg)
    
    # model.preprocess(source= torch.tensor(np.load('/scratch/vb/S2025_TASK3/features/stereo_dev/raw_split/fold3_room12_mix001_deg000_start1978_idx00.npy')))
    lprobs, padding_mask= model.extract_features(source= torch.randn(size= (16, 1, 21, 64)), )
    total_params=sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    breakpoint()
    
    model= MultiheadAttention()
    summary(model= model.to(device), input_size= ((32, 8, 768), (32, 8, 768), (32, 8, 768)), 
        depth= 8, col_names=['input_size', 'output_size', 'num_params'])
    
    model= TransformerSentenceEncoderLayer()
    summary(model= model.to(device), input_size= (32, 8, 768), depth= 8, 
        col_names=['input_size', 'output_size', 'num_params'])
    
    model= TransformerEncoder(cfg= beats_cfg.beats_cfg)
    summary(model= model.to(device), input_size= (32, 8, 768), depth= 10, 
        col_names=['input_size', 'output_size', 'num_params'])
    

   


