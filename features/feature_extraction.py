import os, glob, librosa, yaml, cv2

import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy import signal
from PIL import Image
import nara_wpe.wpe as wpe
from multiprocessing import Pool

import feature_utils

import warnings
warnings.filterwarnings("ignore")

class SELDFeatureExtractor:
    def __init__(self, dataset_cfg, ):
        self.dataset_cfg = dataset_cfg

        # audio feature extraction.
        self.root_dir= self.dataset_cfg['root_dir']
        self.feat_dir= self.dataset_cfg['feat_dir']

        self.sampling_rate = self.dataset_cfg['sampling_rate']
        self.hop_len= int(self.sampling_rate*self.dataset_cfg['hop_len'])
        self.window_len= 2*self.hop_len
        self.n_fft= 2**(self.window_len-1).bit_length()
        self.num_mels= self.dataset_cfg['num_mels']
        self.audio_normalization = self.dataset_cfg['audio_normalization']
        self.eps= 1e-8

        self.split_hop_len= 120
        self.split_window_len= 480
        self.split_n_fft= 2**(self.split_window_len-1).bit_length()

        # video feature extraction.
        self.fps= self.dataset_cfg['fps']

    @staticmethod
    def normalize_audio(x, eps=1e-8):
        max_val= np.max(np.abs(x))
        if max_val > 0:
            x= x / max_val
        return x

    def extract_audio_feature(self, split_type):
        if split_type == 'dev':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split_type == 'eval':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_eval', 'eval-*', '*.wav'))

        feature_path= os.path.join(self.feat_dir, f'stereo_{split_type}')
        os.makedirs(feature_path, exist_ok= True)

        feature_types= ['raw', 'lr_stft', 'lr_logmel', 'foa_iv', 'foa_iv_logmel', 'wy_stft', 'wy_logmel', 'ild', 'ild_logmel', 
            'ipd', 'ipd_logmel', 'direct_w', 'reverb_w', 'direct_stft', 'reverb_stft', 'direct_logmel', 'reverb_logmel', 'drr', 'drr_logmel',
            'stpacc']

        # split_feature_types= []
        for ft in feature_types:
            os.makedirs(os.path.join(feature_path, 'full', ft), exist_ok= True)
            # split_feature_types.append(f'{ft}_split')
            # os.makedirs(os.path.join(feature_path, 'split', f'{ft}_split'), exist_ok= True)

        NUM_PROCESS= 10
        global run_task1
        def run_task1(args):
        # for args in tqdm(list(enumerate(audio_files_path))):
            count, audio_fp= args

            audio_file_name= os.path.splitext(os.path.basename(audio_fp))[0]
            audio_data, sr= load_audio(audio_file= audio_fp, sampling_rate= self.sampling_rate)
            if self.audio_normalization:
                audio_data= self.normalize_audio(audio_data, eps= self.eps)
            
            lr_stft= extract_stft(audio_data= audio_data, n_fft= self.n_fft, hop_len= self.hop_len, window_len= self.window_len)
            lr_logmel= get_Logmel(x= np.abs(lr_stft)**2, sampling_rate= self.sampling_rate, num_mels= self.num_mels)
            np.save(os.path.join(feature_path, 'full', 'lr_stft', f'{audio_file_name}.npy'), lr_stft.transpose(0, 2, 1))
            np.save(os.path.join(feature_path, 'full', 'lr_logmel', f'{audio_file_name}.npy'), lr_logmel.transpose(0, 2, 1))

            wy_stft= extract_wy_stft(audio_data= audio_data, n_fft= self.n_fft, hop_len= self.hop_len, window_len= self.window_len)
            wy_logmel= get_Logmel(x= np.abs(wy_stft)**2, sampling_rate= self.sampling_rate, num_mels= self.num_mels)
            np.save(os.path.join(feature_path, 'full', 'wy_stft', f'{audio_file_name}.npy'), wy_stft.transpose(0, 2, 1))
            np.save(os.path.join(feature_path, 'full', 'wy_logmel', f'{audio_file_name}.npy'), wy_logmel.transpose(0, 2, 1))

            ild, ild_logmel= get_ILD(audio_stft= lr_stft, sampling_rate= self.sampling_rate, num_mels= self.num_mels, eps= self.eps)
            
            np.save(os.path.join(feature_path, 'full', 'ild', f'{audio_file_name}.npy'), ild.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'ild_logmel', f'{audio_file_name}.npy'), ild_logmel.transpose(1, 0))
            
            ipdx, ipdy, ipdx_logmel, ipdy_logmel= get_IPD(audio_stft= lr_stft, sampling_rate= self.sampling_rate, num_mels= self.num_mels)
            np.save(os.path.join(feature_path, 'full', 'ipd', f'{audio_file_name}.npy'), np.stack((ipdx, ipdy), axis= 0).transpose(0, 2, 1))
            np.save(os.path.join(feature_path, 'full', 'ipd_logmel', f'{audio_file_name}.npy'), np.stack((ipdx_logmel, ipdy_logmel), axis= 0).transpose(0, 2, 1))
            
            ivy, ivy_norm, ivy_logmel= get_foa(stft= wy_stft, sampling_rate= self.sampling_rate, num_mels= self.num_mels, eps= self.eps)
            np.save(os.path.join(feature_path, 'full', 'foa_iv', f'{audio_file_name}.npy'), ivy.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'foa_iv_logmel', f'{audio_file_name}.npy'), ivy_logmel.transpose(1, 0))

            L, R= audio_data[0, :], audio_data[1, :]
            W= (L+R)/2

			# extract short-term power of the autocorrelation features.
            stp_acc= extract_stpACC(x= W, nfft= self.n_fft, hop_len= self.hop_len, downsample= True)
            np.save(file= os.path.join(feature_path, 'full', 'stpacc', audio_file_name), arr= stp_acc[None, ...])
            
            direct_stft, reverb_stft, direct_logmel, reverb_logmel, drr, drr_logmel, direct_w, reverb_w= get_distance_feats(audio= audio_data, 
                hop_len= self.hop_len, window_len= self.window_len, n_fft= self.n_fft, sampling_rate= self.sampling_rate, 
                num_mels= self.num_mels, eps= self.eps)
            
            np.save(os.path.join(feature_path, 'full', 'direct_w', f'{audio_file_name}.npy'), direct_w)
            np.save(os.path.join(feature_path, 'full', 'reverb_w', f'{audio_file_name}.npy'), reverb_w)
            # np.save(os.path.join(feature_path, 'full', 'direct_stft', f'{audio_file_name}.npy'), direct_stft.transpose(1, 0))
            # np.save(os.path.join(feature_path, 'full', 'reverb_stft', f'{audio_file_name}.npy'), reverb_stft.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'direct_logmel', f'{audio_file_name}.npy'), direct_logmel)
            np.save(os.path.join(feature_path, 'full', 'reverb_logmel', f'{audio_file_name}.npy'), reverb_logmel)
            # # np.save(os.path.join(feature_path, 'full', 'drr', f'{audio_file_name}.npy'), drr.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'drr_logmel', f'{audio_file_name}.npy'), drr_logmel)
            return
        
        with Pool(NUM_PROCESS) as p:
            p.map(run_task1, tqdm(list(enumerate(audio_files_path))))
            # p.map(run_task1, list(enumerate(audio_files_path)))
        print('Extracting audio feature finished.')
    
    def extract_video_features(self, split_type):
        if split_type=='dev':
            video_files_path= glob.glob(pathname=os.path.join(self.dataset_cfg['root_dir'], 'video_dev', 'dev-*', '*.mp4'))
        elif split_type=='eval':
            video_files_path= glob.glob(pathname=os.path.join(self.dataset_cfg['root_dir'], 'video_eval', 'eval*', '*.mp4'))

        feature_path= os.path.join(self.dataset_cfg['feat_dir'], f'video_{split_type}')
        os.makedirs(feature_path, exist_ok= True)

        feat_types= ['raw', 'raw_split']
        for ft in feat_types:
            os.makedirs(os.path.join(feature_path, ft), exist_ok= True)
		
        NUM_PROCESS= 10
        global run_task2
        def run_task2(args):
		# for args in tqdm(enumerate(video_files_path)):
            count, video_fp= args
            # print(f'Started processing {count} {video_fp.split("/")[-1]}')
            video_file_name= os.path.splitext(os.path.basename(video_fp))[0]

            video_frames= np.array(load_video(video_file= video_fp, video_fps= self.fps))
            np.save(file= os.path.join(feature_path, 'raw', f'{video_file_name}.npy'), arr= video_frames)			
            return
        
        with Pool(NUM_PROCESS) as p:
            p.map(run_task2, tqdm(list(enumerate(video_files_path))))
            # p.map(run_task2, list(enumerate(video_files_path)))
        print('Extracting video feature finished.')

    def extract_labels(self, split_type):
        if split_type=='dev':
            label_files_path= glob.glob(os.path.join(self.dataset_cfg['root_dir'], 'metadata_dev', 'dev-*', '*.csv'))
        elif split_type=='eval':
            label_files_path= glob.glob(os.path.join(self.dataset_cfg['root_dir'], 'metadata_eval', 'eval-*', '*.csv'))

        labels_dir_path= os.path.join(self.dataset_cfg['feat_dir'], f'metadata_{split_type}_adpit')
        os.makedirs(labels_dir_path, exist_ok= True)

        # feat_types= ['label', 'label_split', 'orig_label']
        feat_types= ['label', 'orig_label']
        for ft in feat_types:
            os.makedirs(os.path.join(labels_dir_path, ft), exist_ok= True)

        NUM_PROCESS= 20
        global run_task3
        def run_task3(args):

        # for args in tqdm(list(enumerate(label_files_path))):
            count, label_fp= args
            label_fname= os.path.splitext(os.path.basename(label_fp))[0]
            
            label_data, orig_label_data= feature_utils.load_labels(label_file= label_fp, convert_to_cartesian= True)
            processed_labels= feature_utils.process_labels_adpit(label_data= label_data, 
                label_seq_len= self.dataset_cfg['label_seq_len'], num_classes= self.dataset_cfg['num_classes']).numpy()
            
            np.save(file= os.path.join(labels_dir_path, 'label', f'{label_fname}.npy'), arr= processed_labels)
            np.save(file= os.path.join(labels_dir_path, 'orig_label', f'{label_fname}.npy'), arr= np.array(orig_label_data))
            
            return

        with Pool(NUM_PROCESS) as p:
            p.map(run_task3, tqdm(list(enumerate(label_files_path))))
        #     # p.map(run_task3, list(enumerate(label_files_path)))
        print('Extracting labels finished.')

    def extract_dist_features(self, split_type):
        if split_type == 'dev':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split_type == 'eval':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_eval', 'eval-*', '*.wav'))

        feature_path= os.path.join(self.feat_dir, f'stereo_{split_type}')
        os.makedirs(feature_path, exist_ok= True)

        feature_types= ['direct_stft', 'reverb_stft', 'direct_logmel', 'reverb_logmel', 'drr', 'drr_logmel',
                        'direct_w', 'reverb_w']

        # split_feature_types= []
        for ft in feature_types:
            os.makedirs(os.path.join(feature_path, 'full', ft), exist_ok= True)

        # NUM_PROCESS= 10
        # global run_task4
        # def run_task4(args):
        for args in tqdm(list(enumerate(audio_files_path))):
            count, audio_fp= args
            # print(count, audio_fp)

            audio_file_name= os.path.splitext(os.path.basename(audio_fp))[0]
            
            file1_path= os.path.join(feature_path, 'full', 'direct_logmel', f'{audio_file_name}.npy')
            file2_path= os.path.join(feature_path, 'full', 'reverb_logmel', f'{audio_file_name}.npy')
            file3_path= os.path.join(feature_path, 'full', 'drr_logmel', f'{audio_file_name}.npy')
            
            if (os.path.isfile(file1_path) and os.path.isfile(file2_path) and os.path.isfile(file3_path)):
                continue
            
            audio_data, sr= load_audio(audio_file= audio_fp, sampling_rate= self.sampling_rate)
            if self.audio_normalization:
                audio_data= self.normalize_audio(audio_data, eps= self.eps)

            direct_stft, reverb_stft, direct_logmel, reverb_logmel, drr, drr_logmel, direct_w, reverb_w= get_distance_feats(audio= audio_data,
                hop_len= self.hop_len, window_len= self.window_len, n_fft= self.n_fft, sampling_rate= self.sampling_rate,   
                num_mels= self.num_mels, eps= self.eps)
            
            # np.save(os.path.join(feature_path, 'full', 'direct_stft', f'{audio_file_name}.npy'), direct_stft.transpose(1, 0))
            # np.save(os.path.join(feature_path, 'full', 'reverb_stft', f'{audio_file_name}.npy'), reverb_stft.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'direct_logmel', f'{audio_file_name}.npy'), direct_logmel)
            np.save(os.path.join(feature_path, 'full', 'reverb_logmel', f'{audio_file_name}.npy'), reverb_logmel)
            # np.save(os.path.join(feature_path, 'full', 'drr', f'{audio_file_name}.npy'), drr.transpose(1, 0))
            np.save(os.path.join(feature_path, 'full', 'drr_logmel', f'{audio_file_name}.npy'), drr_logmel)
            np.save(os.path.join(feature_path, 'full', 'direct_w', f'{audio_file_name}.npy'), direct_w)
            np.save(os.path.join(feature_path, 'full', 'reverb_w', f'{audio_file_name}.npy'), reverb_w)
        
            return
        
		# with Pool(NUM_PROCESS) as p:
        #     p.map(run_task4, tqdm(list(enumerate(audio_files_path))))
        #     p.map(run_task4, list(enumerate(audio_files_path)))
        print('Extracting distance-related audio feature finished.')  

    def extract_stpacc(self, split_type):
        if split_type == 'dev':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_dev', 'dev-*', '*.wav'))
        elif split_type == 'eval':
            audio_files_path= glob.glob(pathname= os.path.join(self.root_dir, 'stereo_eval', 'eval-*', '*.wav'))

        feature_path= os.path.join(self.feat_dir, f'stereo_{split_type}')
        os.makedirs(feature_path, exist_ok= True)

        os.makedirs(os.path.join(feature_path, 'full', 'stpacc'), exist_ok= True)

        NUM_PROCESS= 10
        global run_task2
        def run_task2(args):
        # for args in tqdm(list(enumerate(audio_files_path))):
            count, audio_fp= args
            audio_file_name= os.path.splitext(os.path.basename(audio_fp))[0]+'.npy'
            audio_data, sr= load_audio(audio_file= audio_fp, sampling_rate= self.sampling_rate)
            
            L, R= audio_data[0, :], audio_data[1, :]
            W= (L+R)/2

			# extract short-term power of the autocorrelation features.
            stp_acc= extract_stpACC(x= W, nfft= self.n_fft, hop_len= self.hop_len, downsample= True)
            np.save(file= os.path.join(feature_path, 'full', 'stpacc', audio_file_name), arr= stp_acc[None, ...])   
            return
        
        with Pool(NUM_PROCESS) as p:
            p.map(run_task2, tqdm(list(enumerate(audio_files_path))))
            # p.map(run_task2, list(enumerate(video_files_path)))
        print('Extracting STP-ACC feature finished.')

# Audio feature extactor.
def load_audio(audio_file, sampling_rate):
	audio_data, sr= librosa.load(path= audio_file, sr= sampling_rate, mono= False)
	return audio_data, sr    

def extract_stft(audio_data, n_fft, hop_len, window_len, center= True):
	stft= librosa.stft(y= audio_data, n_fft= n_fft, hop_length= hop_len, win_length= window_len, center= center)
	return stft

def extract_wy_stft(audio_data, n_fft, hop_len, window_len, center= True):
	W, Y= (audio_data[0, :]+audio_data[1, :])/2, (audio_data[0, :]-audio_data[1, :])/2
	wy_data= np.concatenate((W[None, :], Y[None, :]), axis= 0)
	wy_stft= extract_stft(audio_data= wy_data, n_fft= n_fft, hop_len= hop_len, window_len= window_len, center= center)
	return wy_stft

def get_Logmel(x, sampling_rate, num_mels):
    mel_spec= librosa.feature.melspectrogram(S= x, sr= sampling_rate, n_mels= num_mels)
    log_mel_spec= librosa.power_to_db(S= mel_spec, ref= 1., amin= 1e-10, top_db= None)
    return log_mel_spec

def get_ILD(audio_stft, sampling_rate, num_mels, eps= 1e-8):
	abs_audio_stft= np.abs(audio_stft)
	ILD= abs_audio_stft[0] / (abs_audio_stft[1] + eps)
	ILD_logmel= get_Logmel(x= ILD**2, sampling_rate= sampling_rate, num_mels= num_mels)
	return ILD, ILD_logmel

def get_IPD(audio_stft, sampling_rate, num_mels):
	phase_diff= np.angle(audio_stft[0]) - np.angle(audio_stft[1])
	IPDx, IPDy= np.cos(phase_diff), np.sin(phase_diff)
	IPDx_logmel= get_Logmel(x= IPDx, sampling_rate= sampling_rate, num_mels= num_mels)
	IPDy_logmel= get_Logmel(x= IPDx, sampling_rate= sampling_rate, num_mels= num_mels)
	return IPDx, IPDy, IPDx_logmel, IPDy_logmel

def get_foa(stft, sampling_rate, num_mels, eps= 1e-8):
	IVy= np.real(stft[1]*np.conj(stft[0]))
	IVy_norm= IVy/(np.abs(stft[0])**2+np.abs(stft[1])**2+eps)
	IVy_logmel= get_Logmel(x= IVy**2, sampling_rate= sampling_rate, num_mels= num_mels)
	return IVy, IVy_norm, IVy_logmel

def get_distance_feats(audio, hop_len, window_len, n_fft, sampling_rate, num_mels, eps= 1e-8, center= True):
    # breakpoint()
    w= (audio[0] + audio[1])/2
    w_stft= extract_stft(audio_data= w, n_fft= n_fft, hop_len= hop_len, window_len= window_len, center= center)
    direct_stft= wpe.wpe(Y= w_stft[:, None, :], taps=60, delay=5, iterations=5, statistics_mode='full').squeeze(axis= 1)
    direct_w= librosa.istft(stft_matrix= direct_stft, hop_length= hop_len, win_length= window_len, 
		n_fft= n_fft, window= 'hann', center= center)
    reverb_w= w- direct_w
    reverb_stft= extract_stft(audio_data= reverb_w, n_fft= n_fft, hop_len= hop_len, window_len= window_len, center= center)
    direct_psd, reverb_psd= np.abs(direct_stft)**2, np.abs(reverb_stft)**2
    direct_psd, reverb_psd= np.clip(direct_psd, a_min= 1e-100, a_max= direct_psd.max()), np.clip(reverb_psd, a_min= 1e-100, a_max= reverb_psd.max())

    direct_logmel= get_Logmel(x= direct_psd.T[..., None], sampling_rate= sampling_rate, num_mels= num_mels)
    reverb_logmel= get_Logmel(x= reverb_psd.T[..., None], sampling_rate= sampling_rate, num_mels= num_mels)
    drr= direct_psd / (reverb_psd + eps)
    drr_logmel= get_Logmel(x= drr.T[..., None], sampling_rate= sampling_rate, num_mels= num_mels)
    return direct_stft, reverb_stft, direct_logmel, reverb_logmel, drr, drr_logmel, direct_w, reverb_w

def extract_stpACC(x, nfft, hop_len, downsample= True, ds_factor= 8):
    x_stft= librosa.stft(y= x, n_fft= nfft, hop_length= hop_len, center= True, 
        window= np.hanning(M= nfft), pad_mode= 'reflect')
    auto_corr= x_stft * np.conj(x_stft)
    
    acc= np.fft.irfft(a= auto_corr.T)
    acc= acc[:, :acc.shape[1]//2]
    acc= np.clip(a= acc, a_min= 1e-100, a_max= acc.max(axis= -1, keepdims= True))
    acc_norm= acc/abs(acc).max(axis= -1, keepdims= True)
    stp_acc= signal.convolve(in1= acc_norm**2, in2= np.hanning(8)[None, :], mode= 'same', method= 'direct')
    stp_acc_norm= stp_acc / stp_acc.max(axis= -1, keepdims= True)
    
    # downsample.
    if downsample:
        stp_acc_norm= signal.resample(x= stp_acc_norm, num= round(stp_acc_norm.shape[1]/ds_factor), axis= -1)
    return stp_acc_norm

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

if __name__=='__main__':
    config_file = './config.yml'
    cfg = feature_utils.get_configurations(config_file= config_file)

    import time
    start_time= time.time()

    dataset_cfg= cfg['DATASET']
    feat_extractor= SELDFeatureExtractor(dataset_cfg= dataset_cfg)
    feat_extractor.extract_labels(split_type= 'dev')
    # feat_extractor.extract_video_features(split_type= 'dev')
    feat_extractor.extract_audio_feature(split_type= 'dev')
    feat_extractor.extract_dist_features(split_type= 'dev')
    # feat_extractor.extract_stpacc(split_type= 'dev')

    end_time= time.time()
    print(f'Total time taken: {(end_time-start_time)/60:.2f} minutes.')
