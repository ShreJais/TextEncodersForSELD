import os, yaml, time
import cv2, warnings
import torch

import numpy as np
import pandas as pd
from PIL import Image

def get_configurations(config_file):
    with open(config_file, 'r') as stream:
        cfg = yaml.safe_load(stream)
    (dataset_cfg, model_cfg, training_cfg, metrics_cfg) = (cfg['DATASET'], cfg['MODEL'], 
				cfg['TRAINING'], cfg['METRICS'])
    return cfg, dataset_cfg, model_cfg, training_cfg, metrics_cfg

def setup(cfg, dataset_cfg, model_cfg):
	"""
	Sets up the environment for training by creating directories for model checkpoints, 
	logging and saving configuration parameters.
	"""
	print(f'Setting the following configuration.')

	# create dir to save model checkpoints.
	model_name= (f'{dataset_cfg["net_type"]}_' + 
		f'{dataset_cfg["modality"]}_{"multiACCDOA" if model_cfg["multiaccdoa"] else "singleACCDOA"}_' + 
			f'{time.strftime("%Y%M%d_%H%M%S")}')
	checkpoint_dir= os.path.join(dataset_cfg['checkpoint_dir'], model_name)
	os.makedirs(checkpoint_dir, exist_ok= True)

	# save the configuration file.
	config_path= os.path.join(checkpoint_dir, 'saved_config.yml')
	with open(config_path, 'w') as f:
		yaml.dump(cfg, f)

	# logging.
	log_dir= os.path.join(checkpoint_dir, dataset_cfg['log_dir'])
	os.makedirs(log_dir, exist_ok= True)
	# initialize the wandb.

	# create the output folder to save the predictions.
	output_dir= os.path.join(checkpoint_dir, dataset_cfg['output_dir'])
	os.makedirs(output_dir, exist_ok= True)

	dir= {"checkpoint_dir": checkpoint_dir, 'log_dir': log_dir, 'output_dir': output_dir}
	return dir

def load_video(video_file, video_fps, resolution= (352, 176)):
	cap= cv2.VideoCapture(video_file)
	frame_interval= max(1, 30 // video_fps) # video_fps =10
	pil_frames, frame_count= [], 0
	while True:
		ret, frame= cap.read()
		if not ret:
			break
		if frame_count % frame_interval == 0:
			# smoothening may not be required since we process the frames individually through resnet.
			resized_frame= cv2.resize(src= frame, dsize= resolution)
			frame_rgb= cv2.cvtColor(src= resized_frame, code= cv2.COLOR_BGR2RGB)
			pil_frame= Image.fromarray(frame_rgb)
			pil_frames.append(pil_frame)
		frame_count+=1
	cap.release()
	return pil_frames

def extract_resnet_features(video_frames, preprocess, backbone, device):
	with torch.no_grad():
		preprocessed_images= [preprocess(image) for image in video_frames]
		preprocessed_images= torch.stack(tensors= preprocessed_images, dim=0).to(device)
		video_feats= backbone(preprocessed_images)
		video_feats= torch.mean(video_feats, dim=1)
		return video_feats
	
def load_labels(label_file, convert_to_cartesian= True):
	orig_label_data= {}

	# with open(label_file, 'r') as file:
	# 	lines= file.readlines()[1:] # skip the header line.
	# 	for line in lines:
	# 		values= line.strip().split(',')
	# 		frame_idx= int(values[0])
	# 		data_row= [int(values[i]) for i in range(1, 6)]
	# 		if frame_idx not in label_data:
	# 			label_data[frame_idx]=[]
	# 		label_data[frame_idx].append(data_row)
	
	df= pd.read_csv(label_file)
	# Group by 'frame' and extract the relevant columns.

	for frame_idx, group in df.groupby('frame'):
		# Convert each row to a list of integers [class, source, azimuth, distance, onscreen]
		data_rows= group[['class', 'source', 'azimuth', 'distance', 'onscreen']].values.tolist()
		orig_label_data[frame_idx]= data_rows

	if convert_to_cartesian:
		label_data= convert_polar_to_cartesian(label_data= orig_label_data)
	return label_data, orig_label_data

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

def process_labels(label_data, label_seq_len, num_classes):
	se_label= torch.zeros(size= (label_seq_len, num_classes)) # se_label.shape: (50, 6, 13)
	x_label= torch.zeros(size= (label_seq_len, num_classes))
	y_label= torch.zeros(size= (label_seq_len, num_classes))
	dist_label= torch.zeros(size= (label_seq_len, num_classes))
	onscreen_label= torch.zeros(size= (label_seq_len, num_classes))

	for frame_idx, active_event_list in label_data.items():
		if frame_idx < label_seq_len:
			for active_event in active_event_list:
				se_label[frame_idx, active_event[0]]= 1
				x_label[frame_idx, active_event[0]]= active_event[2]
				y_label[frame_idx, active_event[0]]= active_event[3]
				dist_label[frame_idx, active_event[0]]= active_event[4]/100
				onscreen_label[frame_idx, active_event[0]]= active_event[5]
	label_mat= torch.cat((se_label, x_label, y_label, dist_label, onscreen_label), dim=1)
	return label_mat

def process_labels_adpit(label_data, label_seq_len, num_classes):	
	se_label= torch.zeros(size= (label_seq_len, 6, num_classes)) # se_label.shape: (50, 6, 13)
	x_label= torch.zeros(size= (label_seq_len, 6, num_classes))
	y_label= torch.zeros(size= (label_seq_len, 6, num_classes))
	dist_label= torch.zeros(size= (label_seq_len, 6, num_classes))
	onscreen_label= torch.zeros(size= (label_seq_len, 6, num_classes))

	for frame_idx, active_event_list in label_data.items():
		if frame_idx < label_seq_len:
			active_event_list.sort(key= lambda x: x[0]) # sort for overlapping from the same class.
			active_event_list_per_class= []
			for i, active_event in enumerate(active_event_list):
				active_event_list_per_class.append(active_event)
				if i == len(active_event_list)-1: # if only class.
					if len(active_event_list_per_class) == 1:
						# No overlapping from the same class.
						active_event_a0= active_event_list_per_class[0]
						se_label[frame_idx, 0, active_event_a0[0]]= 1
						x_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[2]
						y_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[3]
						dist_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[4]/100
						onscreen_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[5]

					elif len(active_event_list_per_class) == 2:
						# Overlapping with 2 sources from the same class.
						active_event_b0= active_event_list_per_class[0]
						se_label[frame_idx, 1, active_event_b0[0]]= 1
						x_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[2]
						y_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[3]
						dist_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[4]/100
						onscreen_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[5]

						active_event_b1= active_event_list_per_class[0]
						se_label[frame_idx, 2, active_event_b1[0]]= 1
						x_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[2]
						y_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[3]
						dist_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[4]/100
						onscreen_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[5]

					else:
						# Overlapping with more than 3 sources from the same class.
						active_event_c0= active_event_list_per_class[0]
						se_label[frame_idx, 3, active_event_c0[0]]= 1
						x_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[2]
						y_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[3]
						dist_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[4]/100
						onscreen_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[5]

						active_event_c1= active_event_list_per_class[0]
						se_label[frame_idx, 4, active_event_c1[0]]= 1
						x_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[2]
						y_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[3]
						dist_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[4]/100
						onscreen_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[5]

						active_event_c2= active_event_list_per_class[0]
						se_label[frame_idx, 5, active_event_c2[0]]= 1
						x_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[2]
						y_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[3]
						dist_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[4]/100
						onscreen_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[5]

				elif active_event[0] != active_event_list[i+1][0]:
					# Next event is not from the same class.
					if len(active_event_list_per_class)==1:
						# No overlapping from the same class.
						active_event_a0= active_event_list_per_class[0]
						se_label[frame_idx, 0, active_event_a0[0]]= 1
						x_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[2]
						y_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[3]
						dist_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[4]/100
						onscreen_label[frame_idx, 0, active_event_a0[0]]= active_event_a0[5]

					elif len(active_event_list_per_class) == 2:
						# Overlapping with 2 sources from the same class.
						active_event_b0= active_event_list_per_class[0]
						se_label[frame_idx, 1, active_event_b0[0]]= 1
						x_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[2]
						y_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[3]
						dist_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[4]/100
						onscreen_label[frame_idx, 1, active_event_b0[0]]= active_event_b0[5]

						active_event_b1= active_event_list_per_class[0]
						se_label[frame_idx, 2, active_event_b1[0]]= 1
						x_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[2]
						y_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[3]
						dist_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[4]/100
						onscreen_label[frame_idx, 2, active_event_b1[0]]= active_event_b1[5]

					else:
						# Overlapping with more than 3 sources from the same class.
						active_event_c0= active_event_list_per_class[0]
						se_label[frame_idx, 3, active_event_c0[0]]= 1
						x_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[2]
						y_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[3]
						dist_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[4]/100
						onscreen_label[frame_idx, 3, active_event_c0[0]]= active_event_c0[5]

						active_event_c1= active_event_list_per_class[0]
						se_label[frame_idx, 4, active_event_c1[0]]= 1
						x_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[2]
						y_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[3]
						dist_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[4]/100
						onscreen_label[frame_idx, 4, active_event_c1[0]]= active_event_c1[5]

						active_event_c2= active_event_list_per_class[0]
						se_label[frame_idx, 5, active_event_c2[0]]= 1
						x_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[2]
						y_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[3]
						dist_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[4]/100
						onscreen_label[frame_idx, 5, active_event_c2[0]]= active_event_c2[5]
					active_event_list_per_class = []

	label_mat= torch.stack((se_label, x_label, y_label, dist_label, onscreen_label), dim=2)
	return label_mat

def organize_labels(labels_dict, max_frames, max_tracks= 10):
	tracks= set(range(max_tracks))
	output_labels_dict= {x: {} for x in range(max_frames)}

	for frame_idx in range(0, max_frames):
		if frame_idx not in labels_dict:
			continue
		for [class_idx, source_idx, azi_angle, dist, onscreen] in labels_dict[frame_idx]:
			if class_idx not in output_labels_dict[frame_idx]:
				output_labels_dict[frame_idx][class_idx]= {}
			if source_idx not in output_labels_dict[frame_idx][class_idx] and source_idx < max_tracks:
				track_idx = source_idx
			else:
				try:
					track_idx= list(set(tracks) - output_labels_dict[frame_idx][class_idx].keys())[0]
				except IndexError:
					warnings.warn(message= 'The number of sources is higher than the number of tracks. Some events will be missed.')
					track_idx= 0 # Overwrite one event.
			output_labels_dict[frame_idx][class_idx][track_idx]= [azi_angle, dist, onscreen]
	return output_labels_dict
