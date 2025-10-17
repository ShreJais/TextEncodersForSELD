import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
from torchinfo import summary

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size= (3, 3), stride= (1, 1), padding= (1, 1), 
        dilation= 1, use_bias= False, pool_size= (2, 2), pool_type= 'avg'):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels= c_in, out_channels= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= dilation, bias= use_bias)
        self.conv2= nn.Conv2d(in_channels= c_out, out_channels= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= dilation, bias= use_bias)
        self.bn1= nn.BatchNorm2d(num_features= c_out)
        self.bn2= nn.BatchNorm2d(num_features= c_out)

        if pool_type=='avg':
            self.pool= nn.AvgPool2d(kernel_size= pool_size)
        elif pool_type=='max':
            self.pool= nn.AvgPool2d(kernel_size= pool_size)
        else:
            raise Exception('pool_type must be avg or max')
        
    def forward(self, x: torch.Tensor):
        x= F.relu(self.bn1(self.conv1(x)))
        x= F.relu(self.bn2(self.conv2(x)))
        x= self.pool(x)
        return x

class ConvBlock2(nn.Module):
    def __init__(self, c_in, c_out, kernel_size= (3, 3), stride= (1, 1), padding= (1, 1), 
        dilation= 1, use_bias= False):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels= c_in, out_channels= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= dilation, bias= use_bias)
        self.conv2= nn.Conv2d(in_channels= c_out, out_channels= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= dilation, bias= use_bias)
        self.bn1= nn.BatchNorm2d(num_features= c_out)
        self.bn2= nn.BatchNorm2d(num_features= c_out)
    def forward(self, x: torch.Tensor):
        x= F.relu(self.bn1(self.conv1(x)))
        x= F.relu(self.bn2(self.conv2(x)))
        return x

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size= (3, 3), stride= (1, 1), padding= (1, 1), 
        dilation= 1, use_bias= False, pool_size= (2, 2), pool_type= 'avg'):
        super().__init__()
        self.conv_block= ConvBlock2(c_in= c_in, c_out= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= dilation, use_bias= use_bias)
        self.res_block= nn.Sequential(
            nn.Conv2d(in_channels= c_in, out_channels= c_out, kernel_size= (1, 1), 
                stride= (1, 1), padding= (0, 0), dilation= 1, bias= use_bias), 
            nn.BatchNorm2d(num_features= c_out))
        if pool_type=='avg':
            self.pool= nn.AvgPool2d(kernel_size= pool_size)
        elif pool_type=='max':
            self.pool= nn.AvgPool2d(kernel_size= pool_size)
        else:
            raise Exception('pool_type must be avg or max')

    def forward(self, x):
        residual= self.res_block(x)
        x= self.conv_block(x)
        x= (x + residual)/math.sqrt(2)
        x= self.pool(x)
        return x

class ResidualCNN8(nn.Module):
    def __init__(self, c_in, c_out= [32, 64, 128, 256], kernel_size= (3, 3), stride= (1, 1), 
        padding= (1, 1), dilation= 1, use_bias= True, pool_size= [2, 2], pool_type= 'avg'):
        super().__init__()
        self.conv_blocks= []
        
        channels= [c_in] + c_out
        for i in range(1, len(channels)):
            self.conv_blocks+= [ResBlock(c_in= channels[i-1], c_out= channels[i], kernel_size= kernel_size, 
                stride= stride, padding= padding, dilation= dilation, use_bias= use_bias, pool_size= pool_size[i-1], 
                pool_type= pool_type)]
        self.conv_blocks= nn.Sequential(*self.conv_blocks)
    
    def forward(self, x):
        x= self.conv_blocks(x)
        return x

class CNN8(nn.Module):
    def __init__(self, c_in, c_out= [32, 64, 128, 256], kernel_size= (3, 3), stride= (1, 1), 
        padding= (1, 1), dilation= 1, use_bias= True, pool_size= [2, 2], pool_type= 'avg'):
        super().__init__()

        if len(pool_size) != len(c_out):
            pool_size= [pool_size]*len(c_out)
        self.conv_blocks= []
        channels= [c_in] + c_out
        for i in range(1, len(channels)):
            self.conv_blocks+= [ConvBlock(c_in= channels[i-1], c_out= channels[i], kernel_size= kernel_size, 
                stride= stride, padding= padding, dilation= dilation, use_bias= use_bias, pool_size= pool_size[i-1], 
                pool_type= pool_type)]
        self.conv_blocks= nn.Sequential(*self.conv_blocks)

    def forward(self, x):
        x= self.conv_blocks(x)
        return x  

class ConvBackbone(nn.Module):
    def __init__(self, cfg, c_in= 1, groups= 1, backbone_type: str= 'CNN8'):
        super().__init__()
        self.cfg= cfg
        self.model_cfg= cfg['backbone'][backbone_type]
        self.scalar= nn.ModuleList([nn.BatchNorm2d(num_features= self.model_cfg['num_mels']) 
            for _ in range(c_in)])
        
        if backbone_type=='CNN8':
            self.conv_blocks= CNN8(
                c_in= c_in, c_out= self.model_cfg['c_out'], kernel_size= self.model_cfg['kernel_size'], 
                stride= self.model_cfg['stride'], padding= self.model_cfg['padding'], 
                dilation= 1, use_bias= self.model_cfg['use_bias'], 
                pool_size= self.model_cfg['pool_size'], 
                pool_type= self.model_cfg['pool_type'])
        
        elif backbone_type=='PatchEmbed':
            self.conv_blocks= nn.Conv2d(
                in_channels= c_in, out_channels= self.model_cfg['c_out'], kernel_size= self.model_cfg['kernel_size'],
                stride= self.model_cfg['kernel_size'], padding= self.model_cfg['padding'], 
                dilation= 1, groups= groups, bias= self.model_cfg['use_bias'])
            
        elif backbone_type=='ResidualCNN8':
            self.conv_blocks= ResidualCNN8(
                c_in= c_in, c_out= self.model_cfg['c_out'], kernel_size= self.model_cfg['kernel_size'], 
                stride= self.model_cfg['stride'], padding= self.model_cfg['padding'], 
                dilation= 1, use_bias= self.model_cfg['use_bias'], 
                pool_size= self.model_cfg['pool_size'], 
                pool_type= self.model_cfg['pool_type'])
    
    def forward(self, x):
        x= x.transpose(1, 3)
        for nch in range(x.shape[-1]):
            x[..., [nch]] = self.scalar[nch](x[..., [nch]])
        x= x.transpose(1, 3)
        # encoder
        x= self.conv_blocks(x).squeeze(2)
        return x

class FeatsEmbeddings(nn.Module):
    def __init__(self, num_feats= 4, embed_dim= 512):
        super().__init__()
        self.num_feats= num_feats
        self.embed_dim= embed_dim

        # learnable embedding for each features.
        self.modality_embedding= nn.Embedding(num_embeddings= num_feats, embedding_dim= self.embed_dim)
        # # learnable positional embedding for each features.
        # self.positional_embedding= nn.Embedding(num_embeddings= self.seq_len, embedding_dim= self.embed_dim)

    def forward(self, feats, modality_id: int= 0):
        batch_size, seq_len, _= feats.shape
        # # get positional embedding.
        # pos_idx= torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(feats.device)
        # pos_embedding= self.positional_embedding(pos_idx)
        
        # get features embedding.
        feats_idx= torch.full(size= (batch_size, seq_len), fill_value= modality_id, dtype= torch.long).to(feats.device)
        feats_embedding= self.modality_embedding(feats_idx)

        feats= feats+ feats_embedding
        return feats

class SeldOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.seld_out= []
        for f_count in range(self.model_cfg['num_ffn_layers']):
            self.seld_out+= [nn.Linear(in_features= self.model_cfg['ffn_indim'], out_features= self.model_cfg['ffn_outdim'], 
                bias= self.model_cfg['use_bias'])]
            # self.seld_out+= [nn.LayerNorm(normalized_shape= self.model_cfg['ffn_dim'])]
            # self.seld_out+= [nn.ReLU()]
        if self.model_cfg['num_ffn_layers']>0:
            self.seld_out+= [nn.Linear(in_features= self.model_cfg['ffn_outdim'], 
                out_features= 3*self.model_cfg['max_polyphony']*self.model_cfg['num_classes'], 
                bias= self.model_cfg['use_bias'])]
        else:
            self.seld_out+= [nn.Linear(in_features= self.model_cfg['ffn_indim'], 
                out_features= 3*self.model_cfg['max_polyphony']*self.model_cfg['num_classes'], 
                bias= self.model_cfg['use_bias'])]
        self.seld_out= nn.Sequential(*self.seld_out)

        self.doa_actfn= nn.Tanh()
        self.dist_actfn= nn.ReLU()
    
    def forward(self, x):
        x= self.seld_out(x)
        x= x.view(x.shape[0], self.model_cfg['max_polyphony'], 3, self.model_cfg['num_classes'])
        doa_preds= self.doa_actfn(x[:, :, 0:2, :])
        dist_preds= self.dist_actfn(x[:, :, 2:3, :])
        preds= torch.cat((doa_preds, dist_preds), dim= -2).unsqueeze(1)
        return preds

class SeldTransformerOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.layer= nn.TransformerDecoderLayer(d_model= self.model_cfg['encoder_dim'], 
            nhead= self.model_cfg['num_attention_heads'], 
            dim_feedforward= self.model_cfg['encoder_dim']*self.model_cfg['ffn_expansion_factor'],
            dropout= self.model_cfg['attention_dropout_rate'], batch_first= True, bias= self.model_cfg['use_bias'])
        self.decoder= nn.TransformerDecoder(decoder_layer= self.layer, num_layers= self.model_cfg['num_decoder_layers'])
        self.seld_out= nn.Linear(in_features= 3*self.model_cfg['encoder_dim'], 
            out_features= 3*self.model_cfg['max_polyphony']*self.model_cfg['num_classes'], 
            bias= self.model_cfg['use_bias'])
        self.doa_actfn= nn.Tanh()
        self.dist_actfn= nn.ReLU()

    def forward(self, target, memory, pos, target_mask= None, memory_mask= None, tgt_key_padding_mask= None, memory_key_padding_mask= None):
        out= self.decoder(tgt= target, memory= memory, tgt_mask= target_mask, memory_mask= memory_mask, 
            memory_key_padding_mask= memory_key_padding_mask, tgt_key_padding_mask= tgt_key_padding_mask)
        audio_token_embeddings= out[range(len(pos[:, 0])), pos[:, 0], :]
        doa_token_embeddings= out[range(len(pos[:, 1])), pos[:, 1], :]
        dist_token_embeddings= out[range(len(pos[:, 2])), pos[:, 2], :]
        out= torch.cat((audio_token_embeddings, doa_token_embeddings, dist_token_embeddings), dim= 1)
        out= self.seld_out(out)
        out= out.view(out.shape[0], self.model_cfg['max_polyphony'], 3, self.model_cfg['num_classes'])
        doa_preds= self.doa_actfn(out[:, :, 0:2, :])
        dist_preds= self.dist_actfn(out[:, :, 2:3, :])
        preds= torch.cat((doa_preds, dist_preds), dim= -2).unsqueeze(1)
        return preds

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        if self.model_cfg['decoder']=='conformer':
            self.decoder= ConformerDecoder(cfg= self.model_cfg['seld_conf']['decoder']['conformer'])
        elif self.model_cfg['decoder']=='transformer':
            self.decoder= TransformerBlocks(cfg= self.model_cfg['seld_conf']['decoder']['transformer'])

    def forward(self, x: torch.Tensor):
        if isinstance(self.decoder, nn.RNNBase):
            x= self.decoder(x)[0]
        else:
            x= self.decoder(x)
        return x

class TransformerBlocks(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.encoder_dim= self.model_cfg['encoder_dim']
        self.positional_encoding= PositionalEncoding(d_model= self.encoder_dim)
        self.layer= nn.TransformerEncoderLayer(d_model= self.encoder_dim, nhead= self.model_cfg['num_attention_heads'], 
            dim_feedforward= self.encoder_dim*self.model_cfg['ffn_expansion_factor'], 
            dropout= self.model_cfg['attention_dropout_rate'], batch_first= True)
        self.decoder= nn.TransformerEncoder(encoder_layer= self.layer, 
            num_layers= self.model_cfg['num_decoder_layers'])

    def forward(self, x):
        batch_size, seq_len, _= x.size()
        pos_embedding= self.positional_encoding(seq_len)
        pos_embedding= pos_embedding.repeat(batch_size, 1, 1)
        x= x+ pos_embedding
        x= self.decoder(x)
        return x
    
# Positional Embedding.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int= 512, max_len: int= 10000):
        super().__init__()
        pe= torch.zeros(size= (max_len, d_model), requires_grad= False)
        pos= torch.arange(start= 0, end= max_len, step= 1, dtype= torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(start= 0, end= d_model, step= 2).float()*(-math.log(10000.)/d_model))
        pe[:, 0::2]= torch.sin(pos*div_term)
        pe[:, 1::2]= torch.cos(pos*div_term)
        pe= pe.unsqueeze(dim=0)
        self.register_buffer('pe', pe) 
        # this ensures 'pe' is part of the model state (moved to GPU, saved with model) but not trainable.

    def forward(self, length: int)-> torch.Tensor:
        return self.pe[:, :length]
    
class ConformerDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.layers= nn.ModuleList(
            [ConformerBlock(cfg= self.model_cfg) for _ in range(self.model_cfg['num_decoder_layers'])])
    
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x= layer(x)
        return x

class ConformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        if self.model_cfg['half_step_residual']:
            self.ff_residual_factor= 0.5
        else:
            self.ff_residual_factor= 1.
        
        self.sequential= nn.Sequential(
            ResidualConnectionModule(module= FeedForwardModule(cfg= self.model_cfg), 
                module_factor= self.ff_residual_factor),
            ResidualConnectionModule(module= MultiHeadSelfAttentionModule(cfg= self.model_cfg), 
                module_factor= 1),
            ResidualConnectionModule(module= ConformerConvModule(cfg= self.model_cfg), 
                module_factor= 1),
            ResidualConnectionModule(module= FeedForwardModule(cfg= self.model_cfg), 
                module_factor= self.ff_residual_factor),
            nn.LayerNorm(normalized_shape= self.model_cfg['encoder_dim'])
        )

    def forward(self, x):
        return self.sequential(x)
    
class ResidualConnectionModule(nn.Module):
    def __init__(self, module: nn.Module, module_factor: float= 1., input_factor: float= 1.):
        super().__init__()
        self.module= module
        self.module_factor= module_factor
        self.input_factor= input_factor
    
    def forward(self, x: torch.Tensor):
        return self.module(x)*self.module_factor + x*self.input_factor

# FeedForward.
class FeedForwardModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.encoder_dim= self.model_cfg['encoder_dim']
        self.ffn_expansion_factor= self.model_cfg['ffn_expansion_factor']
        self.sequential= nn.Sequential(
            nn.LayerNorm(normalized_shape= self.encoder_dim), 
            Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim*self.ffn_expansion_factor, use_bias= self.model_cfg['use_bias']),
            Swish(), 
            nn.Dropout(p= self.model_cfg['ffn_dropout_rate']),
            Linear(in_feats= self.encoder_dim*self.ffn_expansion_factor, out_feats= self.encoder_dim, use_bias= self.model_cfg['use_bias']),
            nn.Dropout(p= self.model_cfg['ffn_dropout_rate']))
    
    def forward(self, x: torch.Tensor):
        return self.sequential(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return x*x.sigmoid()

class Linear(nn.Linear):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_feats, out_feats, use_bias= True):
        super().__init__(in_features= in_feats, out_features= out_feats, bias= use_bias)
        nn.init.xavier_uniform_(tensor= self.weight)
        if use_bias:
            nn.init.zeros_(self.bias)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return super(Linear, self).forward(input= x)

class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.encoder_dim= self.model_cfg['encoder_dim']
        self.positional_encoding= PositionalEncoding(d_model= self.encoder_dim)
        self.layer_norm= nn.LayerNorm(normalized_shape= self.encoder_dim)
        self.attention= RelativeMultiHeadAttention(cfg= self.model_cfg)
        self.dropout= nn.Dropout(p= self.model_cfg['layer_dropout_rate'])

    def forward(self, x, mask= None):
        batch_size, seq_len, _= x.size()
        pos_embedding= self.positional_encoding(seq_len)
        pos_embedding= pos_embedding.repeat(batch_size, 1, 1)
        x= self.layer_norm(x)
        x= self.attention(query= x, key= x, value= x, pos_embedding= pos_embedding, mask= mask)
        return self.dropout(x)

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.encoder_dim= self.model_cfg['encoder_dim']
        self.num_heads= self.model_cfg['num_attention_heads']
        self.head_dim= int(self.encoder_dim // self.num_heads)

        assert self.encoder_dim%self.num_heads==0, 'encoder_dim % num_heads should be zero'

        self.query_proj= Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim, use_bias= self.model_cfg['use_bias'])
        self.key_proj= Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim, use_bias= self.model_cfg['use_bias'])
        self.value_proj= Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim, use_bias= self.model_cfg['use_bias'])
        
        self.out_proj= Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim, use_bias= self.model_cfg['use_bias'])
        self.pos_proj= Linear(in_feats= self.encoder_dim, out_feats= self.encoder_dim, use_bias= False)

        self.dropout= nn.Dropout(p= self.model_cfg['attention_dropout_rate'])
        self.u_bias= nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        self.v_bias= nn.Parameter(torch.Tensor(self.num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.u_bias)
        nn.init.xavier_uniform_(self.v_bias)

    def forward(self, query, key, value, pos_embedding, mask= None):
        # breakpoint()
        batch_size= value.shape[0]
        # query.shape: (batch_size, seq_len, n_heads, head_dim)
        query= self.query_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        # key.shape: (batch_size, n_heads, seq_len, head_dim)
        key= self.key_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # value.shape: (batch_size, n_heads, seq_len, head_dim)
        value= self.value_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # pos_embedding.shape: (batch_size, n_heads, seq_len, head_dim)
        pos_embedding= self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        content_score= torch.matmul(input= (query + self.u_bias).transpose(1, 2), other=key.transpose(2, 3))
        pos_score= torch.matmul(input= (query+self.v_bias).transpose(1, 2), other= pos_embedding.transpose(2, 3))
        pos_score= self._relative_shift(pos_score= pos_score)
        score= (content_score + pos_score)/math.sqrt(self.head_dim)

        if mask is not None:
            mask= mask.unsqueeze(1)
            score.masked_fill_(mask= mask, value=1e-9)
        
        attention_score= F.softmax(input= score, dim= -1)
        attention_score= self.dropout(attention_score) # (batch_size, n_heads, seq_len, seq_len)

        # context.shape: (batch_size, seq_len, n_heads, head_dim)
        context= torch.matmul(input= attention_score, other= value).transpose(1, 2)
        context= context.contiguous().view(batch_size, -1, self.encoder_dim)
        return self.out_proj(context)

    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_len1, seq_len2= pos_score.size()
        zeros= pos_score.new_zeros(batch_size, num_heads, seq_len1, 1)
        padded_pos_score= torch.cat(tensors= [zeros, pos_score], dim=-1)
        padded_pos_score= padded_pos_score.view(batch_size, num_heads, seq_len2+1, seq_len1)
        pos_score= padded_pos_score[:, :, 1:].view_as(pos_score)
        return pos_score

class ConformerConvModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg
        self.encoder_dim= self.model_cfg['encoder_dim']
        assert (self.model_cfg['conv_kernel_size'] - 1) % 2 == 0, 'kernel size should be a odd number for "SAME" padding'
        assert self.model_cfg['conv_expansion_factor']==2, 'Currently, only supports expansion factor 2.'

        self.sequential= nn.Sequential(
            nn.LayerNorm(normalized_shape= self.encoder_dim),
            Transpose(shape= (1, 2)), 
            PointwiseConv1d(c_in= self.encoder_dim, 
                c_out= self.encoder_dim*self.model_cfg['conv_expansion_factor'], stride=1, padding= 0, 
                use_bias= self.model_cfg['use_bias']),
            GLU(dim= 1),
            DepthwiseConv1d(c_in= self.encoder_dim, c_out= self.encoder_dim, 
                kernel_size= self.model_cfg['conv_kernel_size'], stride= 1, 
                padding= (self.model_cfg['conv_kernel_size']-1)//2, use_bias= self.model_cfg['use_bias']),
            nn.BatchNorm1d(num_features= self.encoder_dim),
            Swish(),
            PointwiseConv1d(c_in= self.encoder_dim, c_out= self.encoder_dim, stride= 1, padding= 0, 
                use_bias= self.model_cfg['use_bias']),
            nn.Dropout(p= self.model_cfg['conv_dropout_rate']))
    
    def forward(self, x):
        return self.sequential(x).transpose(1, 2)

class GLU(nn.Module):
    '''
    Introduced in Language Modeling with Gated ConvNets. 
    Helps model complex dependices. Especially useful in sequence models.
    '''
    def __init__(self, dim: int):
        super().__init__()
        self.dim= dim
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        out, gate= x.chunk(2, dim= self.dim)
        return out*gate.sigmoid()

class Transpose(nn.Module):
    """
    Wrapper class of torch.transpose() for sequential module.
    """
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape= shape

    def forward(self, x: torch.Tensor)->torch.Tensor:
        return x.transpose(*self.shape)

class PointwiseConv1d(nn.Module):
    '''
    When kernel size == 1 conv1d, this operation is termed in literature as pointwise convolution. This operation often
    used to match dimension.
    '''
    def __init__(self, c_in, c_out, stride: int= 1, padding: int= 0, use_bias: bool= True):
        super().__init__()
        self.conv= nn.Conv1d(in_channels= c_in, out_channels= c_out, kernel_size= 1, stride= stride, 
            padding= padding, dilation= 1, bias= use_bias)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return self.conv(x)

class DepthwiseConv1d(nn.Module):
    '''
    When groups == in_channels and out_channels == K*in_channels, where k is a positive integer, this operation is termed
    in literature as depthwise convolution.
    '''
    def __init__(self, c_in, c_out, kernel_size, stride: int= 1, padding: int= 0, use_bias: bool= False):
        super().__init__()
        assert c_out%c_in==0, 'out channels should be constant multiple of in channels.'
        self.conv= nn.Conv1d(in_channels= c_in, out_channels= c_out, kernel_size= kernel_size, 
            stride= stride, padding= padding, dilation= 1, groups= c_in, bias= use_bias)
    
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return self.conv(x)
    
import itertools

class FGAEmbedder(nn.Module):
    def __init__(self, act_fn=nn.GELU, in_size= 768*3, out_size= 768):
        super().__init__()
        self.fc1= nn.Linear(in_features= in_size, out_features= in_size)
        self.fc2= nn.Linear(in_features= in_size, out_features= out_size)
        self.act_fn= act_fn()
        self.factor_graph_attention= Attention(util_e= [out_size], pairwise_flag= False)

    def forward(self, x):
        # x is audio embeddings.
        # breakpoint()
        x= self.fc2(self.act_fn(self.fc1(x)))
        attention= self.factor_graph_attention([x])[0]
        return attention

class Attention(nn.Module):
    def __init__(self, util_e, sharing_factor_weights=[], prior_flag=False,
                 sizes=[], size_force=False, pairwise_flag=True,
                 unary_flag=True, self_flag=True):
        """
        The class performs an attention on a given list of utilities representation.
        :param util_e: the embedding dimensions
        :param sharing_factor_weights:  To share weights, provide a dict of tuples:
         {idx: (num_utils, connected utils)
         Note, for efficiency, the shared utils (i.e., history, are connected to ans
          and question only.
         TODO: connections between shared utils
        :param prior_flag: is prior factor provided
        :param sizes: the spatial simension (used for batch-norm and weighted marginalization)
        :param size_force: force spatial size with adaptive avg pooling.
        :param pairwise_flag: use pairwise interaction between utilities
        :param unary_flag: use local information
        :param self_flag: use self interactions between utilitie's entities
        """
        super().__init__()
        self.util_e= util_e
        self.prior_flag = prior_flag
        self.n_utils= len(util_e)
        self.spatial_pool= nn.ModuleDict()
        self.un_models= nn.ModuleList()
        self.self_flag= self_flag
        self.pairwise_flag= pairwise_flag
        self.unary_flag= unary_flag
        self.size_force= size_force

        if len(sizes)==0:
            sizes= [None for _ in self.util_e]
        
        self.sharing_factor_weights= sharing_factor_weights

        # force the provided size
        for idx, e_dim in enumerate(util_e):
            self.un_models.append(Unary(embed_dim= e_dim))
            if self.size_force:
                self.spatial_pool[str(idx)]= nn.AdaptiveAvgPool1d(output_size= sizes[idx])
        
        # Pairwise.
        self.pairwise_models= nn.ModuleDict()

        for ((idx1, e_dim1), (idx2, e_dim2)) in itertools.combinations_with_replacement(enumerate(util_e), 2):
            if self.self_flag and idx1==idx2:
                self.pairwise_models[str(idx1)]= Pairwise(x_embed_dim= e_dim1, x_spatial_dim= sizes[idx1])
            else:
                if self.pairwise_flag:
                    if idx1 in self.sharing_factor_weights:
                        # not connected.
                        if idx2 not in self.sharing_factor_weights[idx1][1]:
                            continue
                    if idx2 in self.sharing_factor_weights:
                        if idx1 not in self.sharing_factor_weights[idx2][1]:
                            continue
                    self.pairwise_models[str((idx1, idx2))]= Pairwise(
                        x_embed_dim= e_dim1, x_spatial_dim= sizes[idx1], y_embed_dim= e_dim2, y_spatial_dim= sizes[idx2])
        # handle reduce potentials (with scalars)
        self.reduce_potentials= nn.ModuleList()
        self.num_potentials= dict()
        self.default_num_potentials= 0

        if self.self_flag:
            self.default_num_potentials+= 1
        if self.unary_flag:
            self.default_num_potentials+= 1
        if self.prior_flag:
            self.default_num_potentials+= 1

        for idx in range(self.n_utils):
            self.num_potentials[idx]= self.default_num_potentials

        # All other utilities.
        if pairwise_flag:
            for idx, (num_utils, connected_utils) in sharing_factor_weights:
                for c_u in connected_utils:
                    self.num_potentials[c_u]+= num_utils
                    self.num_potentials[idx]+= 1
            for k in self.num_potentials:
                if k not in self.sharing_factor_weights:
                    self.num_potentials[k]+= (self.n_utils -1) - len(sharing_factor_weights)
        
        for idx in range(self.n_utils):
            self.reduce_potentials.append(
                module= nn.Conv1d(in_channels= self.num_potentials[idx], out_channels= 1, kernel_size= 1, bias= False))
    
    def forward(self, utils, priors= None):
        assert self.n_utils == len(utils)
        assert (priors is None and not self.prior_flag) or (priors is not None and self.prior_flag and len(priors)==self.n_utils)
        
        batch_size= utils[0].size(0)
        util_factors= dict()
        attention= list()

        if self.size_force:
            for i, (num_utils, _) in self.sharing_factor_weights.items():
                if str(i) not in self.spatial_pool.keys():
                    continue
                else:
                    high_util= utils[i]
                    high_util = high_util.view(num_utils * batch_size, high_util.size(2), high_util.size(3))
                    high_util = high_util.transpose(1, 2)
                    utils[i] = self.spatial_pool[str(i)](high_util).transpose(1, 2)
            
            for i in range(self.n_utils):
                if i in self.sharing_factor_weights or str(i) not in self.spatial_pool.keys():
                    continue
                utils[i]= utils[i].transpose(1, 2)
                utils[i]= self.spatial_pool[str(i)](utils[i]).transpose(1, 2)

                if self.prior_flag and priors[i] is not None:
                    priors[i]= self.spatial_pool[str(i)](priors[i].unsqueeze(1)).squeeze(1)
            
        # handle shared weights.
        for i, (num_utils, connected_list) in self.sharing_factor_weights:
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pairwise_models[str(i)](utils[i]))
            if self.pairwise_flag:
                for j in connected_list:
                    other_util= utils[j]
                    expanded_util= other_util.unsqueeze(1).expand(
                        batch_size, num_utils, other_util.size(1), other_util.size(2)
                        ).contiguous().view(batch_size*num_utils, other_util.size(1), other_util.size(2))

                    if i < j:
                        factor_ij, factor_ji= self.pairwise_models[str((i, j))](utils[i], expanded_util)
                    else:
                        factor_ji, factor_ij= self.pairwise_models[str((i, j))](expanded_util, utils[i])
                    
                    util_factors[i].append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji.view(batch_size, num_utils, factor_ji.size(1)))
        
        # handle local factors.
        for i in range(self.n_utils):
            if i in self.sharing_factor_weights:
                continue
            if self.unary_flag:
                util_factors.setdefault(i, []).append(self.un_models[i](utils[i]))
            if self.self_flag:
                util_factors.setdefault(i, []).append(self.pairwise_models[str(i)](utils[i]))
        
        # joint.
        if self.pairwise_flag:
            for (i, j) in itertools.combinations_with_replacement(range(self.n_utils), 2):
                if i in self.sharing_factor_weights or j in self.sharing_factor_weights:
                    continue
                if i==j:
                    continue
                else:
                    factor_ij, factor_ji= self.pairwise_models[str((i, j))](utils[i], utils[j])
                    util_factors.setdefault(i, []).append(factor_ij)
                    util_factors.setdefault(j, []).append(factor_ji)
        
        # perform attention.
        for i in range(self.n_utils):
            if self.prior_flag:
                if priors[i] is not None:
                    prior= priors[i]  
                else:
                    prior= torch.zeros_like(util_factors[i][0], requires_grad=False).to(util_factors[i][0].device)

                util_factors[i].append(prior)
            
            util_factors[i]= torch.cat([p if len(p.size())==3 else p.unsqueeze(1) for p in util_factors[i]], dim=1)
            util_factors[i]= self.reduce_potentials[i](util_factors[i]).squeeze(i)
            # util_factors[i]= f.softmax(util_factors[i], dim=1).unsqueeze(2)
            util_factors[i]= F.softmax(util_factors[i], dim=1)
            # breakpoint()
            # attention.append(torch.bmm(utils[i].transpose(1, 2), util_factors[i]).squeeze(2))
            attention.append(torch.bmm(utils[i].transpose(1, 2), util_factors[i].transpose(1, 2)).squeeze(2))
        return attention
    
class Unary(nn.Module):
    def __init__(self, embed_dim):
        """
        Captures local entity information.
        """
        super().__init__()
        self.embedding = nn.Conv1d(in_channels= embed_dim, out_channels= embed_dim, kernel_size= 1)
        self.feature_reduction= nn.Conv1d(in_channels= embed_dim, out_channels= 1, kernel_size= 1)

    def forward(self, x):
        # breakpoint()
        x= x.transpose(1, 2) # (B, T, E) --> (B, E, T)
        x_embed= self.embedding(x) # (B, E, T)
        x_nl_embed= F.dropout(F.relu(x_embed), training= self.training) 
        x_potential= self.feature_reduction(x_nl_embed) # (B, 1, T)
        return x_potential.squeeze(1)

class Pairwise(nn.Module):
    def __init__(self, x_embed_dim, x_spatial_dim= None, y_embed_dim= None, y_spatial_dim= None):
        """
            Captures interaction between utilities or entities of the same utility
        :param embed_x_size: the embedding dimension of the first utility
        :param x_spatial_dim: the spatial dimension of the first utility for batch norm and weighted marginalization
        :param embed_y_size: the embedding dimension of the second utility (none for self-interactions)
        :param y_spatial_dim: the spatial dimension of the second utility for batch norm and weighted marginalization
        """
        super().__init__()
        y_embed_dim= y_embed_dim if y_spatial_dim is not None else x_embed_dim
        self.y_spatial_dim= y_spatial_dim if y_spatial_dim is not None else x_spatial_dim
        self.x_spatial_dim= x_spatial_dim

        self.embed_dim= max(x_embed_dim, y_embed_dim)

        self.x_embedding= nn.Conv1d(in_channels= x_embed_dim, out_channels= self.embed_dim, kernel_size= 1)
        self.y_embedding= nn.Conv1d(in_channels= y_embed_dim, out_channels= self.embed_dim, kernel_size= 1)

        if self.x_spatial_dim is not None:
            self.normalize_s= nn.BatchNorm1d(num_features= self.x_spatial_dim*self.y_spatial_dim)
            self.margin_x= nn.Conv1d(in_channels= self.y_spatial_dim, out_channels= 1, kernel_size= 1)
            self.margin_y= nn.Conv1d(in_channels= self.x_spatial_dim, out_channels= 1, kernel_size= 1)

    def forward(self, x, y= None):
        # breakpoint()
        x_t= x.transpose(1, 2) # (b, tx, dx) --> (b, dx, tx)
        y_t= y.transpose(1, 2) if y is not None else x_t # (b, ty, dy) --> (b, dy, ty)

        # project embeddings
        x_embedding= self.x_embedding(x_t) # (b, new_dx, tx)
        y_embedding= self.y_embedding(y_t) # (b, new_dy, tx)
        
        # normalize (cosine-like similarity).
        x_norm= F.normalize(x_embedding, dim=1, ) 
        y_norm= F.normalize(y_embedding, dim=1)

        # pairwise similarity.
        S= x_norm.transpose(1, 2).bmm(y_norm)
        
        # Normalize and marginalize.
        if self.x_spatial_dim is not None:
            S= self.normalize_s(S.view(-1, self.x_spatial_dim*self.y_spatial_dim))
            S= S.view(-1, self.x_spatial_dim, self.y_spatial_dim)

            x_potential= self.margin_x(S.transpose(1, 2)).squeeze(1)
            y_potential= self.margin_y(S).squeeze(1)
        else:
            x_potential= S.mean(dim=2, keepdim= False)
            y_potential= S.mean(dim=1 , keepdim= False)

        if y is None:
            return x_potential
        else:
            return x_potential, y_potential

class GAEmbedder(nn.Module):
    def __init__(self, act_fn=nn.GELU, in_size= 768*3, out_size= 768):
        super().__init__()
        self.fc1= nn.Linear(in_features= in_size, out_features= in_size)
        self.fc2= nn.Linear(in_features= in_size, out_features= out_size)
        self.act_fn= act_fn()
        self.factor_graph_attention= GraphAttention(util_e= [out_size], pairwise_flag= False)

    def forward(self, x):
        # x is audio embeddings.
        # breakpoint()
        x= self.fc2(self.act_fn(self.fc1(x)))
        attention= self.factor_graph_attention([x])[0]
        return attention

class GraphAttention(nn.Module):
    def __init__(self, in_feats, out_feats, dropout_rate= 0.1, alpha=0.2):
        super().__init__()
        self.linear= nn.Linear(in_features= in_feats, out_features= out_feats)       

    def forward(self, x):
        breakpoint()

if __name__=='__main__':
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    import utils
    config_file = './config.yml'
    cfg = utils.get_configurations(config_file= config_file)

    model_cfg= cfg['MODEL']['seld_conf']

    model= CNN8(c_in= 1)

    model= ConvBackbone(cfg= model_cfg, c_in= 1, groups= 1, backbone_type= 'CNN8')

    summary(model= model.to(device), input_size= (4, 1, 250, 64), depth= 4,
        col_names= ['input_size', 'output_size', 'num_params', 'kernel_size'])
    
    model= TransformerBlocks(cfg= model_cfg)

    summary(model= model.to(device), input_size= (4, 1, 250, 64), depth= 4,
        col_names= ['input_size', 'output_size', 'num_params', 'kernel_size'])