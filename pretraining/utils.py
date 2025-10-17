import os, yaml, time, random, warnings
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import torch
from torch.utils.tensorboard import SummaryWriter

# os.environ['TF_CPP_MIN_LOG_LEVEL']= '3'

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# GPU operation have seperate seed.
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # multi_gpu.

def get_configurations(config_file):
	with open(config_file, 'r') as stream:
		cfg = yaml.safe_load(stream)
	return cfg

def setup(cfg, dataset_cfg, model_cfg, training_cfg):
	"""
	Sets up the environment for training by creating directories for model checkpoints, 
	logging and saving configuration parameters.
	"""
	print(f'Setting the following configuration.')
	# create dir to save model checkpoints.
	model_name= (f'{training_cfg["exp_name"]}_' + 
		f'{dataset_cfg["modality"]}_{"multiACCDOA" if model_cfg["multiaccdoa"] else "singleACCDOA"}_' + 
		f'{time.strftime("%Y%m%d_%H%M%S")}')
	checkpoint_dir= os.path.join(dataset_cfg['checkpoint_dir'], model_name)
	os.makedirs(checkpoint_dir, exist_ok= True)

	# save the configuration file.
	config_path= os.path.join(checkpoint_dir, 'saved_config.yml')
	with open(config_path, 'w') as f:
		yaml.dump(cfg, f)

	# logging.
	log_dir= os.path.join(checkpoint_dir, dataset_cfg['log_dir'])
	os.makedirs(log_dir, exist_ok= True)
	
	summary_writer = SummaryWriter(log_dir=str(log_dir))
	# initialize the wandb.

	# create the output folder to save the predictions.
	output_dir= os.path.join(checkpoint_dir, dataset_cfg['output_dir'])
	os.makedirs(output_dir, exist_ok= True)

	dir= {"checkpoint_dir": checkpoint_dir, 'log_dir': log_dir, 'output_dir': output_dir}
	return dir, summary_writer

def get_prompt_template(mode: str= 'random', num_template= 1, num_frames= None):
	prompt_template= []
	if mode=='same':
		temp= 'Sound of {} is perceived in this scene, originating from the {}, around {} away.'
		prompt_template= [temp] * num_template

	elif mode=='random':
		template_options = [
			"The sound of {} comes from direction {}, approximately {} away.",
			"The audio of {} comes from direction {}, approximately {} away.",
			"The noise of {} comes from direction {}, approximately {} away.",
			"The echo of {} comes from direction {}, approximately {} away.",
			"The resonance of {} comes from direction {}, approximately {} away.",
			"The acoustic presence of {} comes from direction {}, approximately {} away.",
			"The auditory event of {} comes from direction {}, approximately {} away.",
			"The recorded sound of {} comes from direction {}, approximately {} away.",
			"The captured audio of {} comes from direction {}, approximately {} away.",
			"The auditory trace of {} comes from direction {}, approximately {} away.",
			"The sonic impression of {} originates from direction {}, nearly {} distant.",
			"The sound of {} emanates from direction {}, about {} away.",
			"The audio of {} arises from direction {}, roughly {} off.",
			"The noise of {} emerges from direction {}, approximately {} away.",
			"The echo of {} is traced to direction {}, nearly {} apart.",
			"The resonance of {} is located at direction {}, about {} distant.",
			"The acoustic presence of {} is positioned at direction {}, close to {}.",
			"The auditory event of {} is detected at direction {}, around {} away.",
			"The recorded sound of {} is perceived from direction {}, about {} distant.",
			"The captured audio of {} is identified from direction {}, approximately {} off.",
			"The auditory trace of {} resonates from direction {}, about {} away.",
			"The sonic impression of {} reverberates from direction {}, nearly {} off.",
			"The sound of {} vibrates from direction {}, roughly {} away.",
			"The audio of {} is carried from direction {}, about {} distant.",
			"The noise of {} travels from direction {}, approximately {} away.",
			"The echo of {} arrives from direction {}, close to {}.",
			"The resonance of {} is directed from direction {}, nearly {} distant.",
			"The acoustic presence of {} projects from direction {}, about {} away.",
			"The auditory event of {} extends from direction {}, around {} away.",
			"The recorded sound of {} is broadcast from direction {}, approximately {}.",
			"The captured audio of {} is localized at direction {}, about {} away.",
			"The auditory trace of {} is pinpointed at direction {}, nearly {} distant.",
			"The sonic impression of {} originates at direction {}, approximately {} away.",
			"The sound of {} is observed at direction {}, about {} off.",
			"The audio of {} is marked at direction {}, roughly {} distant.",
			"The noise of {} is reported from direction {}, about {} away.",
			"The echo of {} is noted at direction {}, nearly {} distant.",
			"The resonance of {} is perceived to arise at direction {}, close to {}.",
			"The acoustic presence of {} is acoustically traced to direction {}, about {} away.",
			"The auditory event of {} is acoustically localized at direction {}, nearly {} distant.",
			"The recorded sound of {} is acoustically identified from direction {}, approximately {} away.",
			"The captured audio of {} is acoustically analyzed as coming from direction {}, about {} off.",
			"The auditory trace of {} is acoustically captured from direction {}, nearly {} distant.",
			"The sonic impression of {} is acoustically registered from direction {}, about {} away.",
			"The sound of {} is acoustically perceived from direction {}, roughly {} distant.",
			"The audio of {} is acoustically noted from direction {}, nearly {} away.",
			"The noise of {} is acoustically observed from direction {}, approximately {} away.",
			"The echo of {} is acoustically recognized from direction {}, nearly {} off.",
			"The resonance of {} is acoustically recorded from direction {}, about {} away.",
			"The acoustic presence of {} is acoustically detected from direction {}, approximately {} distant."] 
		prompt_template+= [random.choice(seq= template_options) for _ in range(num_template)]
	
	if num_frames is not None:
		prompt_templates= []
		for i in range(num_template):
			prompt_templates.extend([prompt_template[i]]*num_frames)
	else:
		prompt_templates= prompt_template
	
	# Get positions of the three '{}' tokens.
	placeholder_indices= []
	prompt_len= []
	for sentence in prompt_templates:
		words= sentence.split()
		idx= [i for i, w in enumerate(words) if '{}' in w]
		placeholder_indices.append(idx)
		prompt_len.append(len(sentence.split()) + 2) # (word count + <sos>/<eos>
			
	return prompt_templates, placeholder_indices, prompt_len

def load_labels(label_file, convert_to_cartesian= True):
	orig_label_data= {}
	df= pd.read_csv(label_file)

	for frame_idx, group in df.groupby('frame'):
		# Convert each row to a list of integers [class, source, azimuth, distance, onscreen]
		data_rows= group[['class', 'source', 'azimuth', 'distance', 'onscreen']].values.tolist()
		orig_label_data[frame_idx]= data_rows
		# max_rows= max(max_rows, len(data_rows))
    
    # breakpoint()
	if convert_to_cartesian:
		label_data= convert_polar_to_cartesian(label_data= orig_label_data)
		return label_data, orig_label_data
	else:
		return orig_label_data, None
	
def convert_polar_to_cartesian(label_data):
	output_label_data= {}
	for frame_idx in label_data.keys():
		if frame_idx not in output_label_data:
			output_label_data[frame_idx]= []
		
		for val in label_data[frame_idx]:
			azimuth_angle_rad= val[2]*np.pi/180
			x, y= np.cos(azimuth_angle_rad), np.sin(azimuth_angle_rad)
			output_label_data[frame_idx].append(val[0:2] + [x, y] + val[3:])
	return output_label_data

def convert_cartesian_to_polar(label_data):
	output_label_data= {}
	for frame_idx in label_data.keys():
		if frame_idx not in output_label_data:
			output_label_data[frame_idx] = []
		for val in label_data[frame_idx]:
			x, y= val[2], val[3]
			azimuth_angle_degree= np.arctan2(y, x)*180/np.pi
			output_label_data[frame_idx].append(val[0:2] + [azimuth_angle_degree] + val[4:])
	return output_label_data

def fold_azimuth_angle(azi_angle):
	'''
	Project the azimuth angle into the range [-90, 90].
	'''
	azi_angle= (azi_angle+180)%360 - 180 # Make sure azi_angle is in range [-180, 180)
	folded_azi_angle= azi_angle.copy()
	# Folding outer ranges [-180, -90) into [-90, 0).
	folded_azi_angle[np.logical_and(-180<=azi_angle, azi_angle<-90)]= -180-azi_angle[np.logical_and(-180<=azi_angle, azi_angle<-90)]
	# Folding outer ranges (90, 180] into (0, 90].
	folded_azi_angle[np.logical_and(90<azi_angle, azi_angle<=180)]= 180-azi_angle[np.logical_and(90<azi_angle, azi_angle<=180)]
	return folded_azi_angle

def write_preds_to_dcase_format(preds, dataset_cfg, metrics_cfg, out_dir, file_list, split= 'dev-test'):
	if not dataset_cfg['multiaccdoa']:
		event_pred, dummy_src_id, x_pred, y_pred, dist_pred, onscreen_pred= get_accdoa_labels(preds= preds, dataset_cfg= dataset_cfg)
		for i in range(event_pred.shape[0]):
			out_dict= get_output_dict_format_single_accdoa(event_pred= event_pred[i].cpu().numpy(), 
				src_id= dummy_src_id[i].cpu().numpy(), x_pred= x_pred[i].cpu().numpy(), 
				y_pred= y_pred[i].cpu().numpy(), dist_pred= dist_pred[i].cpu().numpy(), 
				onscreen_pred= onscreen_pred[i].cpu().numpy(), convert_to_polar= True)
			fname= os.path.basename(file_list[i])[:-3] + '.csv'
			write_to_dcase_output_format(out_dict= out_dict, out_dir= out_dir, file_name= fname, split= split)
	else:
		# event_pred.shape, x_pred.shape, y_pred.shape, dist_pred.shape: (n_tracks, batch_size, n_frames, n_classes)
		event_pred, dummy_src_id, x_pred, y_pred, xy_pred, dist_pred, onscreen_pred= get_multiaccdoa_labels(preds= preds, dataset_cfg= dataset_cfg)
		for i in range(event_pred.shape[1]): 
			out_dict= get_output_dict_format_multi_accdoa(event_pred= event_pred[:, i].cpu().numpy(), src_id= dummy_src_id[:, i].cpu().numpy(),
                x_pred= x_pred[:, i].cpu().numpy(), y_pred= y_pred[:, i].cpu().numpy(), xy_pred= xy_pred[:, i].cpu().numpy(),
                dist_pred= dist_pred[:, i].cpu().numpy(), onscreen_pred= onscreen_pred[:, i].cpu().numpy(), 
                dataset_cfg= dataset_cfg, convert_to_polar= True)
			fname= os.path.basename(file_list[i])[:-3] + '.csv'
			write_to_dcase_output_format(out_dict= out_dict, out_dir= out_dir, file_name= fname, split= split, convert_dist_to_cm= True)

def get_accdoa_labels(preds, dataset_cfg):
	num_classes= dataset_cfg['num_classes']
	x_pred, y_pred= preds[:, :, :num_classes], preds[:, :, num_classes:2*num_classes]
	event_pred= torch.sqrt(x_pred**2+y_pred**2) > 0.5
	dist_pred= preds[:, :, 2*num_classes: 3*num_classes]
	dist_pred[dist_pred<0.]= 0.

	if dataset_cfg['modality']=='audio_visual':
		onscreen_pred= preds[:, :, 3*num_classes:4*num_classes]
	else:
		onscreen_pred= torch.zeros_like(dist_pred, device=dist_pred.device)
	dummy_src_id= torch.zeros_like(dist_pred, device= dist_pred.device)
	return event_pred, dummy_src_id, x_pred, y_pred, dist_pred, onscreen_pred

def get_multiaccdoa_labels(preds, dataset_cfg):
	num_classes= dataset_cfg['num_classes']
	if dataset_cfg['modality']=='audio':
		preds= preds.view(-1, dataset_cfg['label_seq_len'], 3, 3, 13)
		preds= preds.view(preds.shape[0], preds.shape[1], -1)
		x_pred, y_pred, event_pred, dist_pred, xy_pred, onscreen_pred, dummy_src_id= [], [], [], [], [], [], []
		for i in range(dataset_cfg['max_polyphony']):
			x_pred.append(preds[:, :, (3*i+0)*num_classes: (3*i+1)*num_classes])
			y_pred.append(preds[:, :, (3*i+1)*num_classes: (3*i+2)*num_classes])
			event_pred.append(torch.sqrt(x_pred[i]**2+y_pred[i]**2)> 0.5) 
			dist_pred.append(preds[:, :, (3*i+2)*num_classes: (3*i+3)*num_classes])
			dist_pred[i][dist_pred[i]<0.]= 0.
			xy_pred.append(preds[:, :, (3*i+0)*num_classes: (3*i+2)*num_classes])
			dummy_src_id.append(torch.zeros_like(dist_pred[i], device= dist_pred[i].device))
			onscreen_pred.append(torch.zeros_like(dist_pred[i], device= dist_pred[i].device))
		x_pred, y_pred, xy_pred= torch.stack(x_pred, dim=0), torch.stack(y_pred, dim=0), torch.stack(xy_pred, dim=0)
		event_pred, dist_pred= torch.stack(event_pred, dim=0), torch.stack(dist_pred, dim=0)
		dummy_src_id, onscreen_pred= torch.stack(dummy_src_id, dim=0), torch.stack(onscreen_pred, dim=0)
	else:
		x_pred, y_pred, event_pred, dist_pred, xy_pred, onscreen_pred, dummy_src_id= [], [], [], [], [], [], []
		for i in range(dataset_cfg['max_polyphony']):
			x_pred.append(preds[:, :, (4*i+0)*num_classes: (4*i+1)*num_classes])
			y_pred.append(preds[:, :, (4*i+1)*num_classes: (4*i+2)*num_classes])
			event_pred.append(torch.sqrt(x_pred[i]**2+y_pred[i]**2) > 0.5) 
			dist_pred.append(preds[:, :, (4*i+2)*num_classes: (4*i+3)*num_classes])
			dist_pred[i][dist_pred[i]<0.]= 0.
			xy_pred.append(preds[:, :, (4*i+0)*num_classes: (4*i+2)*num_classes])
			dummy_src_id.append(torch.zeros_like(dist_pred[i], device= dist_pred[i].device))
			onscreen_pred.append(preds[:, :, (4*i+3)*num_classes: (4*i+4)*num_classes])
		x_pred, y_pred, xy_pred= torch.stack(x_pred, dim=0), torch.stack(y_pred, dim=0), torch.stack(xy_pred, dim=0)
		event_pred, dist_pred= torch.stack(event_pred, dim=0), torch.stack(dist_pred, dim=0)
		dummy_src_id, onscreen_pred= torch.stack(dummy_src_id, dim=0), torch.stack(onscreen_pred, dim=0)
	return event_pred, dummy_src_id, x_pred, y_pred, xy_pred, dist_pred, onscreen_pred

def get_output_dict_format_single_accdoa(event_pred, src_id, x_pred, y_pred, dist_pred, onscreen_pred, convert_to_polar= True):
	out_dict= {}
	for frame_count in range(event_pred.shape[0]):
		for class_count in range(event_pred.shape[1]):
			if event_pred[frame_count][class_count] > 0.5:
				if frame_count not in out_dict:
					out_dict[frame_count]= []
				if convert_to_polar:
					azi_rad= np.arctan2(y_pred, x_pred)
					azi_deg= azi_rad*180/np.pi
					out_dict[frame_count].append([class_count, src_id[frame_count][class_count], 
						azi_deg[frame_count][class_count], dist_pred[frame_count][class_count], 
						onscreen_pred[frame_count][class_count]])
				else:
					out_dict[frame_count].append([class_count, src_id[frame_count][class_count], 
						x_pred[frame_count][class_count], y_pred[frame_count][class_count], 
						dist_pred[frame_count][class_count], onscreen_pred[frame_count][class_count]])
	return out_dict

def get_output_dict_format_multi_accdoa(event_pred, src_id, x_pred, y_pred, xy_pred, dist_pred, 
	onscreen_pred, dataset_cfg, convert_to_polar= True):
	out_dict= {}
	
	for frame_count in range(event_pred.shape[1]):
		for class_count in range(event_pred.shape[-1]):
			similarity_flags= determine_similar_locs(event_pred= event_pred[:, frame_count, class_count], 
				x_pred= x_pred[:, frame_count, class_count], y_pred= y_pred[:, frame_count, class_count], dataset_cfg= dataset_cfg)
			if sum(list(similarity_flags.values()))==0:
				for i in range(dataset_cfg['max_polyphony']):
					if event_pred[i, frame_count, class_count] > 0.5:
						if frame_count not in out_dict:
							out_dict[frame_count]=[]
						if convert_to_polar:
							azi_rad= np.arctan2(y_pred[i, frame_count, class_count], x_pred[i, frame_count, class_count])
							azi_deg= azi_rad*180/np.pi
							out_dict[frame_count].append([class_count, src_id[i, frame_count, class_count], 
								azi_deg, dist_pred[i, frame_count, class_count], onscreen_pred[i, frame_count, class_count]])
						else:
							out_dict[frame_count].append([class_count, src_id[i, frame_count, class_count], 
								x_pred[i, frame_count, class_count], y_pred[i, frame_count, class_count], 
								dist_pred[i, frame_count, class_count], onscreen_pred[i, frame_count, class_count]])				
			elif sum(list(similarity_flags.values()))==1:
				if frame_count not in out_dict:
					out_dict[frame_count]= []
				pairs= {'01': [2, [0, 1]], '12': [0, [1, 2]], '20': [1, [2, 0]]}
				for key, vals in pairs.items():
					val1, val2= vals
					if similarity_flags[key]:
						if event_pred[val1, frame_count, class_count] > 0.5:
							if convert_to_polar:
								azi_rad= np.arctan2(y_pred[val1, frame_count, class_count], x_pred[val1, frame_count, class_count])
								azi_deg= azi_rad*180/np.pi
								out_dict[frame_count].append([class_count, src_id[val1, frame_count, class_count], 
									azi_deg, dist_pred[val1, frame_count, class_count], onscreen_pred[val1, frame_count, class_count]])
							else:
								out_dict[frame_count].append([class_count, src_id[val1, frame_count, class_count], 
									x_pred[val1, frame_count, class_count], y_pred[val1, frame_count, class_count], 
									dist_pred[val1, frame_count, class_count], onscreen_pred[val1, frame_count, class_count]])
						
						x_pred_fc= (x_pred[val2[0], frame_count, class_count]+x_pred[val2[1], frame_count, class_count])/2
						y_pred_fc= (y_pred[val2[0], frame_count, class_count]+y_pred[val2[1], frame_count, class_count])/2
						dist_pred_fc= (dist_pred[val2[0], frame_count, class_count]+dist_pred[val2[1], frame_count, class_count])/2
						onscreen_pred_fc= onscreen_pred[val2[0], frame_count, class_count]
						src_id_fc= src_id[val2[0], frame_count, class_count]
						if convert_to_polar:
							azi_rad= np.arctan2(y_pred_fc, x_pred_fc)
							azi_deg= azi_rad*180/np.pi
							out_dict[frame_count].append([class_count, src_id_fc, azi_deg, dist_pred_fc, onscreen_pred_fc])
						else:
							out_dict[frame_count].append([class_count, src_id_fc, x_pred_fc, y_pred_fc, dist_pred_fc, onscreen_pred_fc])
			elif sum(list(similarity_flags.values()))>=2:
				if frame_count not in out_dict:
					out_dict[frame_count]= []
				x_pred_fc= x_pred[:, frame_count, class_count].sum() / 3
				y_pred_fc= y_pred[:, frame_count, class_count].sum() / 3
				dist_pred_fc= dist_pred[:, frame_count, class_count].sum() / 3
				onscreen_pred_fc= onscreen_pred[0, frame_count, class_count]
				src_id_fc= src_id[0, frame_count, class_count]

				if convert_to_polar:
					azi_rad= np.arctan2(y_pred_fc, x_pred_fc)
					azi_deg= azi_rad*180/np.pi
					out_dict[frame_count].append([class_count, src_id_fc, azi_deg, dist_pred_fc, onscreen_pred_fc])
				else:
					out_dict[frame_count].append([class_count, src_id_fc, x_pred_fc, y_pred_fc, dist_pred_fc, onscreen_pred_fc])
	return out_dict

def determine_similar_locs(event_pred, x_pred, y_pred, dataset_cfg):
	pairs= {'01': [0, 1], '12': [1, 2], '20': [2, 0]}
	flags= {}
	for key, val in pairs.items():
		if (event_pred[val[0]]==1) and (event_pred[val[1]]==1):
			dist_bw_coord= dist_bw_cartesian_coord(x1= x_pred[val[0]], x2= x_pred[val[1]], y1= y_pred[val[0]], y2= y_pred[val[1]])
			flags[key]= 1 if dist_bw_coord < dataset_cfg['thresh_unify'] else 0
		else:
			flags[key]= 0
	return flags

def dist_bw_cartesian_coord(x1, x2, y1, y2):
	"""
	Angular distance between 2 cartesian coordinates.
	"""
	n1, n2= np.sqrt(x1**2+y1**2+1e-10), np.sqrt(x2**2+y2**2+1e-10)
	x1, y1, x2, y2= x1/n1, y1/n1, x2/n2, y2/n2
	# compute the distance.
	dist= x1*x2 + y1*y2
	dist= np.clip(dist, a_min=-1, a_max=1)
	dist= np.arccos(dist)*180/np.pi
	return dist

def write_to_dcase_output_format(out_dict, out_dir, file_name, split, convert_dist_to_cm= True):
	os.makedirs(os.path.join(out_dir, split), exist_ok= True)
	file_path= os.path.join(out_dir, split, file_name)
	with open(file_path, 'w') as f:
		f.write('frame,class,source,azimuth,distance,onscreen\n')
		for frame_idx, values in out_dict.items():
			for value in values:
				azi_angle_rounded= round(float(value[2]))
				dist_rounded= round(float(value[3]*100)) if convert_dist_to_cm else round(float(value[3]))
				f.write(f'{int(frame_idx)},{int(value[0])},{int(value[1])},{azi_angle_rounded},{dist_rounded},{int(value[4])}\n')
