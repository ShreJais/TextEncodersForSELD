import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

import model_utils, clipseg, beats
import transformers
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

import warnings
warnings.filterwarnings("ignore")

class SELD_AudioPretraining(nn.Module):
	def __init__(self, model_cfg):
		
		super().__init__()
		
		# configurations file
		self.model_cfg = model_cfg
		self.backbon_cin= self.model_cfg['backbone_inch']
		
		case= self.model_cfg['case']

		if self.model_cfg['audio_backbone']=='BEATS':
			self.beats_cfg= beats.BEATSConfig(cfg= self.model_cfg['seld_conf']['backbone']['BEATS'])
			self.audio_backbone= beats.BEATS(cfg= self.beats_cfg)
			audio_backbone_ckpts= torch.load(f= self.model_cfg['pretrain']['audio_backbone'])
			# load weights.
			self.audio_backbone.load_state_dict(audio_backbone_ckpts['model'])
			self.audio_backbone.predictor= None
			self.audio_backbone.requires_grad_(False)
			
			# interpolate pretrained weights. One experiments can be done without interpolation weights.
			interpolated_pretrained_weights= F.interpolate(self.audio_backbone.patch_embedding.weight.data, size= (5, 16), mode= 'bilinear', align_corners= False)
			patch_embedding= nn.Conv2d(1, 512, kernel_size= (5, 16), stride= (5, 16))
			patch_embedding.weight.data= interpolated_pretrained_weights
			self.audio_backbone.patch_embedding= patch_embedding
			self.audio_backbone.patch_embedding.requires_grad_(True)
			self.audio_backbone.post_extract_proj.requires_grad_(True)

		else:
			self.audio_backbone= model_utils.ConvBackbone(
				cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['audio'], groups= 1, backbone_type= self.model_cfg['audio_backbone'])
			# attention across time.
			self.audio_transformer= model_utils.ConformerDecoder(cfg= self.model_cfg['seld_conf']['conformer'])

		
		self.audio_ivspec_backbone= model_utils.ConvBackbone(
			cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['audio_iv'], groups= 1, backbone_type= self.model_cfg['doa_backbone'])
		self.dist_backbone= model_utils.ConvBackbone(
			cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['dist'], groups= 1, backbone_type= self.model_cfg['distance_backbone'])

		# attention across time.
		self.audio_iv_transformer= model_utils.ConformerDecoder(cfg= self.model_cfg['seld_conf']['conformer'])
		self.dist_transformer= model_utils.ConformerDecoder(cfg= self.model_cfg['seld_conf']['conformer'])

		self.feats_modality_embedding= model_utils.FeatsEmbeddings(num_feats= self.model_cfg['num_feats'], embed_dim= self.model_cfg['embed_dim'])
		
		# attention transformers in different frequency features.
		self.decoder= model_utils.Decoder(cfg= self.model_cfg)

		self.audio_proj= model_utils.FGAEmbedder(act_fn= nn.GELU, 
			in_size= self.model_cfg['fga_conf'][self.model_cfg['audio_proj']]['input_size'], 
			out_size= self.model_cfg['fga_conf'][self.model_cfg['audio_proj']]['output_size'])
		self.doa_proj= model_utils.FGAEmbedder(act_fn= nn.GELU, 
			in_size= self.model_cfg['fga_conf'][self.model_cfg['doa_proj']]['input_size'], 
			out_size= self.model_cfg['fga_conf'][self.model_cfg['doa_proj']]['output_size'])
		self.dist_proj= model_utils.FGAEmbedder(act_fn= nn.GELU, 
			in_size= self.model_cfg['fga_conf'][self.model_cfg['distance_proj']]['input_size'], 
			out_size= self.model_cfg['fga_conf'][self.model_cfg['distance_proj']]['output_size'])
		
		self.seld_output= model_utils.SeldOutput(cfg= self.model_cfg['seld_conf']['output']['ffn'])

	def forward(self, audio_specs, audio_ivspecs, dist_specs):
		if self.model_cfg['audio_backbone']=='BEATS':
			audio_specs= self.audio_backbone.extract_features(source= audio_specs)[0]
		else:
			audio_specs= self.audio_backbone(audio_specs)
			batch_size, nch, n_frames, nfreq_feats= audio_specs.shape
			audio_specs= audio_specs.transpose(1, 3).reshape(-1, n_frames, nch)
			audio_specs= self.audio_transformer(audio_specs)
			audio_specs= audio_specs.reshape(batch_size, -1, n_frames, nch).transpose(1, 2).reshape(-1, nfreq_feats, nch)
		
		audio_ivspecs= self.audio_ivspec_backbone(audio_ivspecs)
		dist_specs= self.dist_backbone(dist_specs)

		batch_size, nch, n_frames, nfreq_feats= audio_ivspecs.shape
		audio_ivspecs= audio_ivspecs.transpose(1, 3).reshape(-1, n_frames, nch) # (batch_size*nfreq_feats, n_frames, nch)
		dist_specs= dist_specs.transpose(1, 3).reshape(-1, n_frames, nch) # (batch_size*nfreq_feats, n_frames, nch)
		
		audio_ivspecs= self.audio_iv_transformer(audio_ivspecs).reshape(batch_size, -1, n_frames, nch).transpose(1, 2).reshape(-1, nfreq_feats, nch)
		dist_specs= self.dist_transformer(dist_specs).reshape(batch_size, -1, n_frames, nch).transpose(1, 2).reshape(-1, nfreq_feats, nch)
		
		if self.model_cfg['audio_backbone']=='BEATS':
			audio_specs= audio_specs.view(batch_size, nfreq_feats, n_frames, -1)
			audio_specs= audio_specs.transpose(1, 2)
			audio_specs= audio_specs.reshape(batch_size*n_frames, nfreq_feats, -1)

		if self.model_cfg['num_feats']==2:
			audio_ivspecs= self.feats_modality_embedding(audio_ivspecs, modality_id= 0)		
			dist_specs= self.feats_modality_embedding(dist_specs, modality_id= 1)
			fused_specs= torch.cat((audio_ivspecs, dist_specs), dim= 1)
			fused_specs= self.decoder(fused_specs)
			audio_ivspecs, dist_specs= fused_specs.chunk(chunks= 2, dim= 1)
		else:
			audio_specs= self.feats_modality_embedding(audio_specs, modality_id= 0)
			audio_ivspecs= self.feats_modality_embedding(audio_ivspecs, modality_id= 1)
			dist_specs= self.feats_modality_embedding(dist_specs, modality_id= 2)
			fused_specs= torch.cat((audio_specs, audio_ivspecs, dist_specs), dim= 1)
			fused_specs= self.decoder(fused_specs)
			audio_specs, audio_ivspecs, dist_specs= fused_specs.chunk(chunks= 3, dim= 1)
		
		audio_token_embeddings= self.audio_proj(audio_specs)
		doa_token_embeddings= self.doa_proj(audio_ivspecs)
		dist_token_embeddings= self.dist_proj(dist_specs)

		fused_feats= torch.cat((audio_token_embeddings, doa_token_embeddings, dist_token_embeddings), dim= -1)

		seld_preds= self.seld_output(fused_feats)
		seld_preds= seld_preds.reshape(batch_size, n_frames, seld_preds.shape[2], seld_preds.shape[3], seld_preds.shape[4])
		return seld_preds

if __name__=='__main__':
	device= 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f'Using device: {device}')

	import utils
	config_file= 'config.yml'
	cfg= utils.get_configurations(config_file= config_file)
	model_cfg= cfg['MODEL']
	
	model= SELD_AudioPretraining(model_cfg= model_cfg)
	input_data= [torch.randn(16, 1, 251, 64).to(device), torch.randn(16, 4, 251, 64).to(device), 
				 torch.randn(16, 5, 251, 64).to(device)]
				 
	summary(model= model.to(device), input_data= input_data, depth= 1, 
		col_names= ['input_size', 'output_size', 'num_params', 'trainable'])
