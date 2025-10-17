import os, glob

import warnings
warnings.filterwarnings(action= 'ignore')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SeldAudioDataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()
        self.dataset_cfg= cfg['DATASET']
        self.model_cfg= cfg['MODEL']
        self.training_cfg= cfg['TRAINING']

        if mode == 'dev_train':
            self.folds= self.training_cfg['dev_train_folds']
            self.mode= 'dev'
            self.feats_folder= 'train'
        elif mode == 'dev_test':
            self.folds= self.training_cfg['dev_test_folds']
            self.feats_folder= 'test'
            self.mode= 'dev'
        else:
            raise ValueError(f'Invalid mode: {mode}. Choose from [dev_train, dev_test].')
        self.feat_dir= os.path.join('/ssd_scratch/vb/S2025_TASK3/features', f'{self.feats_folder}_full')

        # file path for audio and video.
        self.files_list= sorted(self.get_file_list())
        
        self.audio_feats_type= self.dataset_cfg['audio_feats_type']
        self.dist_feats_type= self.dataset_cfg['dist_feats_type']

    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, index):
        file_name= self.files_list[index]
        feats= torch.load(os.path.join(self.feat_dir, file_name), weights_only= False)
        
        # get features.
        audio_spec= torch.from_numpy(feats['audio_spec'])
        iv_spec= torch.from_numpy(feats['iv'])
        ild_spec= torch.from_numpy(feats['ild'])
        ipd_spec= torch.from_numpy(feats['ipd'])
        direct_spec= torch.from_numpy(feats['dlogmel'])
        reverb_spec= torch.from_numpy(feats['rlogmel'])
        drr_spec= torch.from_numpy(feats['drrlogmel'])
        stp_acc= torch.from_numpy(feats['stpacc']).float()

        target_label= torch.from_numpy(feats['target_label'])
        orig_label= feats['orig_label'].tolist()
        orig_label= torch.from_numpy(self.update_original_label(orig_label= orig_label))

        idx= torch.randint(low= 0, high= 2, size= (1,))
        event_spec= audio_spec[idx]
        audio_iv_spec= torch.cat((audio_spec, iv_spec[None], ild_spec[None]), dim= 0)

        if self.model_cfg['case']==0:
            dist_spec= torch.cat((direct_spec, reverb_spec, drr_spec), dim= -1).permute(2, 0, 1)
            dist_spec= torch.cat((ipd_spec, dist_spec), dim= 0)
        elif self.model_cfg['case']==1:
            dist_spec= torch.cat((direct_spec, reverb_spec), dim= -1).permute(2, 0, 1)
            dist_spec= torch.cat((ipd_spec, dist_spec), dim= 0)
        elif self.model_cfg['case']==2:
            dist_spec= drr_spec.permute(2, 0, 1)
            dist_spec= torch.cat((ipd_spec, dist_spec), dim= 0)
        elif self.model_cfg['case']==3:
            dist_spec= torch.cat((direct_spec, reverb_spec), dim= -1).permute(2, 0, 1)
        elif self.model_cfg['case']==4:
            dist_spec= torch.cat((ipd_spec, stp_acc), dim= 0)
        elif self.model_cfg['case']==5:
            dist_spec= torch.cat((direct_spec, reverb_spec), dim= -1).permute(2, 0, 1)
            dist_spec= torch.cat((dist_spec, stp_acc), dim= 0)

        return event_spec, audio_iv_spec, dist_spec, target_label, orig_label, file_name

    def get_file_list(self):
        files_path= glob.glob(os.path.join(self.feat_dir, '*.pt'))
        base_files= [os.path.basename(f) for f in files_path]
        return base_files
    
    def update_original_label(self, orig_label):
        final_label= np.ones((self.dataset_cfg['label_seq_len'], 6, 5))*-1
        for idx in range(self.dataset_cfg['label_seq_len']):
            if idx in orig_label.keys():
                final_label[idx, :len(orig_label[idx])]= orig_label[idx]
        return final_label
    
def get_dataloader(cfg, mode):
    dataset= SeldAudioDataset(cfg= cfg, mode= mode)
    if mode=='dev_train':
        batch_size, shuffle= cfg['TRAINING']['train_batch_size'], True
    else:
        batch_size, shuffle= cfg['TRAINING']['test_batch_size'], False

    dataloader= DataLoader(dataset= dataset, batch_size= batch_size, shuffle= shuffle, num_workers= cfg['TRAINING']['num_workers'], 
        drop_last= False)
    return dataloader

if __name__=='__main__':
    import utils
    config_file= './config.yml'
    cfg= utils.get_configurations(config_file)
    train_dataset= SeldAudioDataset(cfg= cfg, mode= 'dev_train')	

    for i in range(len(train_dataset)):
        batch= train_dataset[i]
        break

    train_loader= get_dataloader(cfg= cfg, mode= 'dev_train')
    print(f'len(train_loader): {len(train_loader)}')

    from tqdm import tqdm

    for batch in tqdm(train_loader):
        print(f'event_specs.shape: {batch[0].shape}')
        print(f'audio_iv_specs.shape: {batch[1].shape}')
        print(f'dist_specs.shape: {batch[2].shape}')
        print(f'target_labels.shape: {batch[3].shape}')
        print(f'orig_labels.shape: {batch[4].shape}')
        break

    val_loader= get_dataloader(cfg= cfg, mode= 'dev_test')
    print(f'len(val_loader): {len(val_loader)}')
    

