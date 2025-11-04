import os, glob
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from multiprocessing import Pool
from tqdm import tqdm
import torch

class SaveDataset:
	def __init__(self, cfg, mode= 'dev_train'):
		super().__init__()
		self.dataset_cfg= cfg['DATASET']
		self.model_cfg= cfg['MODEL']
		self.training_cfg= cfg['TRAINING']

		if mode=='dev_train':
			self.folds= self.training_cfg['dev_train_folds']
			self.mode= 'dev'
			self.feat_folder= 'train'

		elif mode=='dev_test':
			self.folds= self.training_cfg['dev_test_folds']
			self.mode= 'dev'
			self.feat_folder= 'test'

		else:
			raise ValueError(f'Invalid mode: {mode}. Choose from [dev_train, dev_test].')

		self.files_list= sorted(self.get_file_list())
		self.audio_feats_type= self.dataset_cfg['audio_feats_type']
		self.dist_feats_type= self.dataset_cfg['dist_feats_type']

		self.create_combined_dataset()

	def get_file_list(self):
		files_path= []
		for fold in self.folds:
			files_path+= glob.glob(os.path.join(self.dataset_cfg['feat_dir'], 'video_dev', 'raw', f'{fold}*.npy'))
		base_files= [os.path.basename(f) for f in files_path]
		return base_files
	
	def create_combined_dataset(self):
		scratch_output_dir= os.path.join(self.dataset_cfg['feat_dir'], f'{self.feat_folder}_full')
		ssd_scratch_output_dir= os.path.join('/ssd_scratch/sj/S2025_TASK3/features', f'{self.feat_folder}_full')
		os.makedirs(scratch_output_dir, exist_ok= True)
		os.makedirs(ssd_scratch_output_dir, exist_ok= True)
		
		NUM_PROCESS= 10
		global run_task3
		def run_task3(args):
			count, file= args
			
			features_labels= {}
			features_labels['imgs']= np.load(os.path.join(self.dataset_cfg['feat_dir'], f'video_{self.mode}', 'raw', file))
			for key, val in self.audio_feats_type.items():
				features_labels[key]= np.load(os.path.join(self.dataset_cfg['feat_dir'], f'stereo_{self.mode}', 'full', val, file))
			for key, val in self.dist_feats_type.items():
				features_labels[key]= np.load(os.path.join(self.dataset_cfg['feat_dir'], f'stereo_{self.mode}', 'full', val, file))

			# add labels.
			features_labels['target_label']= np.load(os.path.join(self.dataset_cfg['feat_dir'], f'metadata_{self.mode}_adpit', 'label', file))
			features_labels['orig_label']= np.load(os.path.join(self.dataset_cfg['feat_dir'], f'metadata_{self.mode}_adpit', 'orig_label', file), allow_pickle= True)

			torch.save(features_labels, os.path.join(scratch_output_dir, file.replace('.npy', '.pt')))
			torch.save(features_labels, os.path.join(ssd_scratch_output_dir, file.replace('.npy', '.pt')))
			return
		
		with Pool(NUM_PROCESS) as p:
			p.map(run_task3, tqdm(list(enumerate(self.files_list))))
		# 	# p.map(run_task3, list(enumerate(self.files_list)))
		print('Saving dataset finished.')

if __name__=='__main__':
	import feature_utils
	config_file = './config.yml'
	cfg= feature_utils.get_configurations(config_file)
	save_train_dataset= SaveDataset(cfg= cfg, mode= 'dev_train')
	save_test_dataset= SaveDataset(cfg= cfg, mode= 'dev_test')
