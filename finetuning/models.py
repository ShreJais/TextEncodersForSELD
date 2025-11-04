import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary
import os, random

from pretraining import model_utils, beats
import clipseg
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

		self.get_tokens= True

		if self.model_cfg['audio_backbone']=='BEATS':
			self.beats_cfg= beats.BEATSConfig(cfg= self.model_cfg['seld_conf']['backbone']['BEATS'])
			self.audio_backbone= beats.BEATS(cfg= self.beats_cfg)
			audio_backbone_ckpts= torch.load(f= self.model_cfg['pretrain']['audio_backbone'])
			# load weights.
			self.audio_backbone.load_state_dict(audio_backbone_ckpts['model'])
			self.audio_backbone.predictor= None
			self.audio_backbone.requires_grad_(False)
			# breakpoint()
			interpolated_pretrained_weights= F.interpolate(self.audio_backbone.patch_embedding.weight.data, size= (5, 16), mode= 'bilinear', align_corners= False)
			patch_embedding= nn.Conv2d(1, 512, kernel_size= (5, 16), stride= (5, 16))
			patch_embedding.weight.data= interpolated_pretrained_weights
			self.audio_backbone.patch_embedding= patch_embedding
			self.audio_backbone.patch_embedding.requires_grad_(True)
			self.audio_backbone.post_extract_proj.requires_grad_(True)

		else:
			self.audio_backbone= model_utils.ConvBackbone(
				cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['audio'], groups= 1, backbone_type= self.model_cfg['audio_backbone'])
			if self.model_cfg['time_attention']=='transformer':
				self.audio_transformer= model_utils.TransformerBlocks(cfg= self.model_cfg['seld_conf']['transformer'])
			elif self.model_cfg['time_attention']=='conformer':
				self.audio_transformer= model_utils.ConformerDecoder(cfg= self.model_cfg['seld_conf']['conformer'])
		
		self.audio_ivspec_backbone= model_utils.ConvBackbone(
			cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['audio_iv'], groups= 1, backbone_type= self.model_cfg['doa_backbone'])
		self.dist_backbone= model_utils.ConvBackbone(
			cfg= self.model_cfg['seld_conf'], c_in= self.backbon_cin[case]['dist'], groups= 1, backbone_type= self.model_cfg['distance_backbone'])
		
		if self.model_cfg['time_attention']=='transformer':
			self.audio_iv_transformer= model_utils.TransformerBlocks(cfg= self.model_cfg['seld_conf']['transformer'])
			self.dist_transformer= model_utils.TransformerBlocks(cfg= self.model_cfg['seld_conf']['transformer'])
		
		elif self.model_cfg['time_attention']=='conformer':
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
			audio_specs= self.audio_backbone.extract_features(source= audio_specs, normalize= self.model_cfg['audio_normalization'])[0]
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

		if self.get_tokens:
			return audio_token_embeddings, doa_token_embeddings, dist_token_embeddings
		else:
			fused_feats= torch.cat((audio_token_embeddings, doa_token_embeddings, dist_token_embeddings), dim= -1)
			seld_preds= self.seld_output(fused_feats)
			seld_preds= seld_preds.reshape(batch_size, n_frames, seld_preds.shape[2], seld_preds.shape[3], seld_preds.shape[4])
			return seld_preds

class SELDNet(nn.Module):
	def __init__(self, pretrained_model_cfg, model_cfg):
		super().__init__()
		# configurations file
		self.pretrained_model_cfg = pretrained_model_cfg
		self.model_cfg = model_cfg

		self.source_info= {0: 'Woman', 1: 'Man', 2: 'Clap', 3: 'Phone', 4: 'Laugh', 5: 'Household', 
			6: 'Footsteps', 7: 'Door', 8: 'Music', 9: 'Instrument', 10: 'Tap', 11: 'Bell', 12: 'Knock', -1: 'None'}

		self.pretrained_encoder= SELD_AudioPretraining(model_cfg= self.pretrained_model_cfg)

		# load pretrained weights based on best pretrained fscore model or best pretrained loss model.
		if self.pretrained_model_cfg['pretrain']['model_type']=='fscore':
			pretrained_weights_path= os.path.join(self.pretrained_model_cfg['pretrain']['seld'], 'best_fscore_model', 'best_model.pt')
		elif self.pretrained_model_cfg['pretrain']['model_type']=='loss':
			pretrained_weights_path= os.path.join(self.pretrained_model_cfg['pretrain']['seld'], 'best_loss_model', 'best_model.pt')
		else:
			raise ValueError(f'Model type {self.model_cfg["pretrain"]["model_type"]} not supported.')
		pretrained_weights= torch.load(pretrained_weights_path, weights_only= False)
		
		self.pretrained_encoder.load_state_dict(pretrained_weights['seld_model'])

		if self.model_cfg['text_encoder']=='CLIP':
			# text tokenizer for placeholders.
			self.tokenizer= transformers.AutoTokenizer.from_pretrained(
				pretrained_model_name_or_path= 'CIDAS/clipseg-rd64-refined', cache_dir= '/scratch/sj')

			# key, val: 'Sound of "AUdio information" is perceived in this scene, originating from the "Doa information", around "distance info" away.'
			# query: 'Sound of "<AUDIO>" is perceived in this scene, originating from the "<DOA>", around "<DIST>" away, and it is <ONSCREEN>.'

			# Add a new special tokens to the tokenizer.
			special_tokens= {"additional_special_tokens": ["<AUDIO>", "<DOA>", "<DIST>", "<NONE>", "<ONSCREEN>"]}
			self.tokenizer.add_special_tokens(special_tokens)
			self.doa_id= self.tokenizer.convert_tokens_to_ids('<DOA>')
			self.dist_id= self.tokenizer.convert_tokens_to_ids('<DIST>')
			self.audio_id= self.tokenizer.convert_tokens_to_ids('<AUDIO>')
			self.none_id= self.tokenizer.convert_tokens_to_ids('<NONE>')

			# self.source_token_info= {key: self.tokenizer(val, return_tensors= 'pt', add_special_tokens= False, return_attention_mask= False)['input_ids'] for key, val in self.source_info.items()}
			
			self.grounder= clipseg.CLIPSegmentation.from_pretrained(
				pretrained_model_name_or_path= 'CIDAS/clipseg-rd64-refined', cache_dir= '/scratch/sj')
			
			# Since the tokenizer has a bigger vocab size than the CLIP text embedding table.
			# So if we don't resize the embedding table, token_ids will contain 
			# values > embedding.num_embedings --> CUDA CRASH.

			# This expands the clip.text_model.embeddings.token_embedding so it matches the new vocab size.
			# It creates random inital embeddings for the new tokens.
			# self.av_grounder.clip.resize_token_embeddings(len(self.tokenizer)) # Can't use it directly.

			# we have to manually resize the num_embeddings
			self.grounder.resize_token_embeddings(new_vocab_size= len(self.tokenizer))
			self.grounder.requires_grad_(False)

		elif self.model_cfg['text_encoder']=='BERT':
			self.tokenizer= transformers.AutoTokenizer.from_pretrained(
				pretrained_model_name_or_path= 'google/bert_uncased_L-12_H-512_A-8', cache_dir= '/scratch/sj')
			# Add a new special tokens to the tokenizer.
			special_tokens= {"additional_special_tokens": ["<AUDIO>", "<DOA>", "<DIST>", "<NONE>", "<ONSCREEN>"]}
			self.tokenizer.add_special_tokens(special_tokens)
			self.doa_id= self.tokenizer.convert_tokens_to_ids('<DOA>')
			self.dist_id= self.tokenizer.convert_tokens_to_ids('<DIST>')
			self.audio_id= self.tokenizer.convert_tokens_to_ids('<AUDIO>')
			self.none_id= self.tokenizer.convert_tokens_to_ids('<NONE>')

			self.grounder= transformers.BertModel.from_pretrained(
				pretrained_model_name_or_path= 'google/bert_uncased_L-12_H-512_A-8', cache_dir= '/scratch/sj')
			self.grounder.resize_token_embeddings(new_num_tokens= len(self.tokenizer))
			self.grounder.requires_grad_(False)
		
		if self.model_cfg['seld_out']=='ffn1':
			self.seld_output= model_utils.SeldOutput(cfg= self.model_cfg['seld_conf']['output']['ffn1'])

		elif self.model_cfg['seld_out']=='ffn2':
			self.seld_output= self.pretrained_encoder.seld_output
			
		elif self.model_cfg['seld_out']=='transformer':
			self.seld_output= model_utils.SeldTransformerOutput(cfg= self.model_cfg['seld_conf']['output']['transformer'])

	def forward(self, audio_specs, audio_ivspecs, dist_specs, encoded_tokens, pos):
		audio_token_embeddings, doa_token_embeddings, dist_token_embeddings= self.pretrained_encoder(audio_specs= audio_specs, 
			audio_ivspecs= audio_ivspecs, dist_specs= dist_specs)
		batch_size= audio_specs.shape[0]
		n_frames= audio_token_embeddings.shape[0]//batch_size
		
		if self.model_cfg['text_encoder']=='CLIP':
			input_embeddings= self.grounder.clip.text_model.embeddings.token_embedding(encoded_tokens['input_ids']).to(audio_token_embeddings.dtype)
			if self.model_cfg['seld_out']=='transformer':
				query_embeddings= input_embeddings.clone()

			input_embeddings[range(len(pos[:, 0])), pos[:, 0], :]= audio_token_embeddings.squeeze(1)
			input_embeddings[range(len(pos[:, 1])), pos[:, 1], :]= doa_token_embeddings.squeeze(1)
			input_embeddings[range(len(pos[:, 2])), pos[:, 2], :]= dist_token_embeddings.squeeze(1)

			position_ids= torch.arange(encoded_tokens['input_ids'].shape[-1]).to(encoded_tokens['input_ids'].device)
			position_embeddings= self.grounder.clip.text_model.embeddings.position_embedding(position_ids).to(input_embeddings.dtype)

			input_embeddings= input_embeddings + position_embeddings
			
			causal_attention_mask= self.grounder._build_causal_attention_mask(batch_size= batch_size*n_frames, 
				seq_len= encoded_tokens['input_ids'].shape[-1], dtype= input_embeddings.dtype).to(input_embeddings.device)

			if encoded_tokens['attention_mask'] is not None:
				attn_mask= _prepare_4d_attention_mask(mask= encoded_tokens['attention_mask'], dtype= input_embeddings.dtype)
			else:
				attn_mask= None	
			encoder_outputs= self.grounder.clip.text_model.encoder(inputs_embeds= input_embeddings, attention_mask= attn_mask, 
				causal_attention_mask= causal_attention_mask.unsqueeze(1), output_attentions= False, output_hidden_states= False, return_dict= True)

			last_hidden_state= self.grounder.clip.text_model.final_layer_norm(encoder_outputs['last_hidden_state']).to(input_embeddings.dtype)
			eot_pos= encoded_tokens['attention_mask'].sum(dim= -1) - 1
			pooled_output= last_hidden_state[range(len(eot_pos)), eot_pos, :]
			final_embeddings= self.grounder.clip.text_projection(pooled_output)

		elif self.model_cfg['text_encoder']=='BERT':
			input_embeddings= self.grounder.embeddings.word_embeddings(encoded_tokens['input_ids']).to(audio_token_embeddings.dtype)
			input_embeddings[range(len(pos[:, 0])), pos[:, 0], :]= audio_token_embeddings.squeeze(1)
			input_embeddings[range(len(pos[:, 1])), pos[:, 1], :]= doa_token_embeddings.squeeze(1)
			input_embeddings[range(len(pos[:, 2])), pos[:, 2], :]= dist_token_embeddings.squeeze(1)

			position_ids= torch.arange(encoded_tokens['input_ids'].shape[-1]).to(encoded_tokens['input_ids'].device)
			position_embeddings= self.grounder.embeddings.position_embeddings(position_ids).to(input_embeddings.dtype)
			segment_embeddings= self.grounder.embeddings.token_type_embeddings(encoded_tokens['token_type_ids']).to(input_embeddings.dtype)

			input_embeddings= input_embeddings + position_embeddings + segment_embeddings
			
			if encoded_tokens['attention_mask'] is not None:
				attn_mask= _prepare_4d_attention_mask(mask= encoded_tokens['attention_mask'], dtype= input_embeddings.dtype)
			else:
				attn_mask= None
			encoder_outputs= self.grounder.encoder(hidden_states= input_embeddings, attention_mask= attn_mask, 
				head_mask= None, output_attentions= False, output_hidden_states= False, return_dict= True)
			last_hidden_state= encoder_outputs.last_hidden_state
			final_embeddings= encoder_outputs.last_hidden_state[:, 0, :]
			# final_embeddings= self.grounder.pooler(cls_output.unsqueeze(1)) # other way to get the pooled output.

		if self.model_cfg['seld_out']=='ffn1':
			seld_preds= self.seld_output(final_embeddings)
			seld_preds= seld_preds.reshape(batch_size, n_frames, seld_preds.shape[2], seld_preds.shape[3], seld_preds.shape[4])
			return final_embeddings, seld_preds
		
		elif self.model_cfg['seld_out']=='ffn2':
			audio_token_embeddings= last_hidden_state[range(len(pos[:, 0])), pos[:, 0], :]
			doa_token_embeddings= last_hidden_state[range(len(pos[:, 1])), pos[:, 1], :]
			dist_token_embeddings= last_hidden_state[range(len(pos[:, 2])), pos[:, 2], :]
			fused_feats= torch.cat((audio_token_embeddings, doa_token_embeddings, dist_token_embeddings), dim= -1)
			seld_preds= self.seld_output(fused_feats)
			seld_preds= seld_preds.reshape(batch_size, n_frames, seld_preds.shape[2], seld_preds.shape[3], seld_preds.shape[4])
			return final_embeddings, seld_preds

		elif self.model_cfg['seld_out']=='transformer':
			seld_preds= self.seld_output(target= query_embeddings, memory= final_embeddings.unsqueeze(1), 
				tgt_key_padding_mask= (encoded_tokens['attention_mask']==0), pos= pos)
			seld_preds= seld_preds.reshape(batch_size, n_frames, seld_preds.shape[2], seld_preds.shape[3], seld_preds.shape[4])
			return final_embeddings, seld_preds

	def get_placeholder_tokens(self, prompt_texts):
		if isinstance(prompt_texts, str):
			prompt_texts= [prompt_texts]
		
		replaced_prompts= [p.format("<AUDIO>", "<DOA>", "<DIST>") for p in prompt_texts]
		encoded_tokens= self.tokenizer(replaced_prompts, return_tensors= 'pt', 
			padding= 'max_length', truncation= True, max_length= 77, )
		pos= []
		for token_ids in encoded_tokens['input_ids']:
			pos_audio= (token_ids == self.audio_id).nonzero(as_tuple= True)[0].item()
			pos_doa= (token_ids == self.doa_id).nonzero(as_tuple= True)[0].item()
			pos_dist= (token_ids == self.dist_id).nonzero(as_tuple= True)[0].item()
			pos.append([pos_audio, pos_doa, pos_dist])
		return encoded_tokens, pos
	
	def get_original_label_embeddings(self, orig_labels, encoded_tokens, pos):
		batch_size, seq_len = encoded_tokens['input_ids'].shape
		device = orig_labels.device
		orig_labels= orig_labels.view(-1, orig_labels.shape[-2], orig_labels.shape[-1])
		
		# for the case of no sound descriptions.
		no_sound_descriptions = [
			"This scene does not contain any noticeable sounds.", "No audible sources are detected in the environment.",
			"The recording is silent without distinct events.", "No meaningful audio activity is present in this scene.",
			"There are no perceivable sound events here.", "The scene is free of identifiable sounds.",
			"No clear sound sources are audible.", "The audio contains no specific sound cues.", 
			"Nothing distinct can be heard in this scene.", "No relevant acoustic events are perceived."]

		# few special tokens to join the labels.
		resp_id= torch.as_tensor(self.tokenizer('respectively', add_special_tokens=False, return_attention_mask=False)['input_ids']).to(device)
		comma_id= self.tokenizer(",", add_special_tokens=False, return_attention_mask=False)['input_ids']
		and_id= self.tokenizer("and", add_special_tokens=False, return_attention_mask=False)['input_ids']

		for i in range(batch_size):
			ol = orig_labels[i]
			num_labels = (ol[:, 0] != -1).sum().item()
			if num_labels == 0:
				text_string= random.choice(no_sound_descriptions)
				text_tokens= self.tokenizer(text_string, return_tensors= 'pt', padding= 'max_length', truncation= True, max_length= 77)
				encoded_tokens['input_ids'][i]= text_tokens['input_ids']
				encoded_tokens['attention_mask'][i]= text_tokens['attention_mask']
			else:
				# This can be improved to get better original label embedding.
				source_tokens, doa_tokens, dist_tokens= [], [], []
				for j in range(num_labels):
					# add source tokens
					source_tokens.append(self.tokenizer(self.source_info[ol[j, 0].item()], add_special_tokens=False, return_attention_mask=False)['input_ids'])
					# source_tokens.append([self.source_token_info[ol[j, 0].item()].item()])
					# add doa tokens.
					doa_tokens.append(self.tokenizer(str(ol[j, 2].item()) + 'degrees', add_special_tokens=False, return_attention_mask=False)['input_ids'])
					# add distance tokens.
					dist_tokens.append(self.tokenizer(str(ol[j, 3].item()/100)+'meters', add_special_tokens=False, return_attention_mask=False)['input_ids'])
					if num_labels > 1:
						# add comma token if not the last one.
						if j < num_labels - 2:
							source_tokens.append(comma_id)
							doa_tokens.append(comma_id)
							dist_tokens.append(comma_id)
						elif j == num_labels - 2:
							source_tokens.append(and_id)
							doa_tokens.append(and_id)
							dist_tokens.append(and_id)
				
				source_tokens= torch.as_tensor(sum(source_tokens, []), device=device).view(-1)
				doa_tokens= torch.as_tensor(sum(doa_tokens, []), device=device).view(-1)
				dist_tokens= torch.as_tensor(sum(dist_tokens, []), device=device).view(-1)
				source_tokens, doa_tokens, dist_tokens= source_tokens.to(device), doa_tokens.to(device), dist_tokens.to(device)
				
				if num_labels > 1:
					added_token_length= len(source_tokens) + len(doa_tokens) + len(dist_tokens) + len(resp_id)	
					curr_token, curr_token_len= encoded_tokens['input_ids'][i], (encoded_tokens['attention_mask'][i]==1).sum()
					encoded_tokens['input_ids'][i]= torch.cat((curr_token[:pos[i][0]], source_tokens, curr_token[pos[i][0]+1:pos[i][1]], 
						doa_tokens, curr_token[pos[i][1]+1: pos[i][2]], dist_tokens, curr_token[pos[i][2]+1: curr_token_len-2], 
						resp_id, curr_token[curr_token_len-2:]))[:seq_len]
					encoded_tokens['attention_mask'][i][:curr_token_len+added_token_length-3]= 1
				else:
					added_token_length= len(source_tokens) + len(doa_tokens) + len(dist_tokens)
					curr_token, curr_token_len= encoded_tokens['input_ids'][i], (encoded_tokens['attention_mask'][i]==1).sum()
					encoded_tokens['input_ids'][i]= torch.cat((curr_token[:pos[i][0]], source_tokens, curr_token[pos[i][0]+1:pos[i][1]], 
						doa_tokens, curr_token[pos[i][1]+1: pos[i][2]], dist_tokens, curr_token[pos[i][2]+1:]))[:seq_len]
					encoded_tokens['attention_mask'][i][:curr_token_len+added_token_length-3]= 1

		if self.model_cfg['text_encoder']=='CLIP':
			gt_embeds= self.grounder.clip.text_model.embeddings.token_embedding(encoded_tokens['input_ids'])
			position_ids= torch.arange(seq_len).to(device)
			position_embeddings = self.grounder.clip.text_model.embeddings.position_embedding(position_ids)
			hidden_states= gt_embeds + position_embeddings
			causal_attention_mask = self.grounder._build_causal_attention_mask(batch_size=batch_size, 
				seq_len=seq_len, dtype=hidden_states.dtype).to(device)
    
			if encoded_tokens['attention_mask'] is not None:
				attn_mask= _prepare_4d_attention_mask(mask= encoded_tokens['attention_mask'], dtype= hidden_states.dtype)
			else:
				attn_mask= None
    
			encoder_outputs= self.grounder.clip.text_model.encoder(inputs_embeds= hidden_states, attention_mask=attn_mask, 
				causal_attention_mask=causal_attention_mask.unsqueeze(1), output_attentions=False, output_hidden_states=False, return_dict=True)

			last_hidden_state= self.grounder.clip.text_model.final_layer_norm(encoder_outputs['last_hidden_state'])
    
			eot_pos= encoded_tokens['attention_mask'].sum(dim=-1) - 1
			pooled_output= last_hidden_state[range(len(eot_pos)), eot_pos, :]
			# final groun-truth embeddings for L_{embed} loss.
			finalgt_embeddings = self.grounder.clip.text_projection(pooled_output)

		elif self.model_cfg['text_encoder']=='BERT':
			gt_embeds= self.grounder.embeddings.word_embeddings(encoded_tokens['input_ids'])
			position_ids= torch.arange(seq_len).to(device)
			position_embeddings = self.grounder.embeddings.position_embeddings(position_ids)
			hidden_states= gt_embeds + position_embeddings
			if encoded_tokens['attention_mask'] is not None:
				attn_mask= _prepare_4d_attention_mask(mask= encoded_tokens['attention_mask'], dtype= hidden_states.dtype)
			else:
				attn_mask= None
			encoder_outputs= self.grounder.encoder(hidden_states= hidden_states, attention_mask= attn_mask, 
				head_mask= None, output_attentions= False, output_hidden_states= False, return_dict= True)
			last_hidden_state= encoder_outputs.last_hidden_state
			cls_output= encoder_outputs.last_hidden_state[:, 0, :]
			# final groun-truth embeddings for L_{embed} loss.
			finalgt_embeddings= self.grounder.pooler(cls_output.unsqueeze(1))
		
		return finalgt_embeddings