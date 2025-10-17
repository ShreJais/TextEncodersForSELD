import os
import numpy as np

import torch
from multiprocessing import Pool
from tqdm import tqdm
import utils

class SeldMetrics:
    def __init__(self, metrics_cfg, num_classes= 13):
        self.metrics_cfg= metrics_cfg
        self.num_classes= num_classes

        # Variables for location-sensitive detection preformance.
        self.true_pos= np.zeros(self.num_classes)
        self.false_pos= np.zeros(self.num_classes)
        self.false_pos_spatial= np.zeros(self.num_classes)
        self.false_neg= np.zeros(self.num_classes)

        self.num_ref= np.zeros(self.num_classes)

        self.doaT= self.metrics_cfg['doa_threshold']
        self.distT= np.inf if self.metrics_cfg['dist_threshold']=='inf' else 0
        self.reldistT= self.metrics_cfg['reldist_threshold']
        self.req_onscreen= self.metrics_cfg['req_onscreen']

        self.S, self.D, self.I= 0, 0, 0 

        # Variables for class-sensitive localization performance.
        self.total_doa_error= np.zeros(self.num_classes)
        self.total_dist_error= np.zeros(self.num_classes)
        self.total_reldist_error= np.zeros(self.num_classes)
        self.total_onscreen_correct= np.zeros(self.num_classes)

        self.de_true_pos= np.zeros(self.num_classes)
        self.de_false_pos= np.zeros(self.num_classes)
        self.de_false_neg= np.zeros(self.num_classes)

        assert self.metrics_cfg['average'] in ['micro', 'macro'], "Only 'micro' and 'macro' average are supported."

    def compute_seld_scores(self):
        """
        Collect the final SELD scores
        :return: returns both location-sensitive detection scores and class-sensitive localization scores:
            mean_results, and classwise results
        """
        eps= np.finfo(float).eps 
        class_wise_results= []

        if self.metrics_cfg['average'] == 'micro':
            pass
        elif self.metrics_cfg['average'] == 'macro':
            # Location-sensitive detection performance.
            fscore= self.true_pos/(eps + self.true_pos + self.false_pos_spatial + 0.5*(self.false_pos+self.false_neg))

            # Class-sensitive localization performance.
            doa_error= self.total_doa_error/(self.de_true_pos+eps)
            doa_error[self.de_true_pos==0]= np.nan
            dist_error= self.total_dist_error/(self.de_true_pos+eps)
            dist_error[self.de_true_pos==0]= np.nan
            reldist_error= self.total_reldist_error/(self.de_true_pos+eps)
            reldist_error[self.de_true_pos==0]= np.nan
            onscreen_acc= self.total_onscreen_correct/(self.de_true_pos+eps)
            onscreen_acc[self.de_true_pos==0]= np.nan

            class_wise_results= np.array([fscore, doa_error, dist_error, reldist_error, onscreen_acc])
            mean_fscore, mean_doa_error= fscore.mean(), np.nanmean(doa_error)
            mean_reldist_error, mean_dist_error, mean_onscreen_acc= np.nanmean(reldist_error), np.nanmean(dist_error), np.nanmean(onscreen_acc)

            mean_results= {'fscore': mean_fscore, 'doa_error': mean_doa_error, 'dist_error': mean_dist_error,
                           'reldist_error': mean_reldist_error, 'onscreen_acc': mean_onscreen_acc}
        else:
            raise NotImplementedError('Only micro and macro averaging are supported.')
        
        return mean_results, class_wise_results
    
    def update_seld_scores(self, preds, labels):
        """
        Computes the SELD scores given a prediction and ground truth labels.
            :param pred: dictionary containing the predictions for every frame
                pred[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
            :param gt: dictionary containing the ground truth for every frame
                gt[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        """
        eps= np.finfo(float).eps
        
        for frame_count in range(len(labels.keys())):
            loc_false_neg, loc_false_pos= 0, 0

            for class_count in range(self.num_classes):
                # Counting the number of reference tracks for each class.
                num_gt_doas= len(labels[frame_count][class_count]) if class_count in labels[frame_count] else None
                num_pred_doas= len(preds[frame_count][class_count]) if class_count in preds[frame_count] else None

                if num_gt_doas is not None:
                    self.num_ref[class_count] += num_gt_doas
                if class_count in labels[frame_count] and class_count in preds[frame_count]:
                    # True positive.
                    # Note: For multiple tracks per class, associate the predicted DOAs to 
                    #       corresponding reference DOA-tracks using hungarian algorithm on 
                    #       the azimuth estimation and then compute the average spatial
                    #       distance b/w the associated reference-predicted tracks.
                    gt_values= np.array(list(labels[frame_count][class_count].values()))
                    gt_az_angle, gt_dist, gt_onscreen= gt_values[:, 0], gt_values[:, 1], gt_values[:, 2]
                    pred_values= np.array(list(preds[frame_count][class_count].values()))
                    pred_az_angle, pred_dist, pred_onscreen= pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]
                    
                    # Reference and predicted track matching.
                    doa_error, row_idx, col_idx= utils.least_distance_between_gt_pred(gt_az_angle= gt_az_angle, pred_az_angle= pred_az_angle)
                    dist_error= np.abs(gt_dist[row_idx] - pred_dist[col_idx])
                    reldist_error= dist_error/(gt_dist[row_idx] + eps)
                    onscreen_correct= (gt_onscreen[row_idx]==pred_onscreen[col_idx])

                    Pc= len(pred_az_angle) # Pc= num_predevents_class_c
                    Rc= len(gt_az_angle) # Rc= num_refevents_class_c
                    FNc= max(0, Rc-Pc) # FNc= false_neg_class_c
                    FPcinf= max(0, Pc-Rc) # FPcinf= false_pos_class_c
                    Kc= min(Pc, Rc)
                    TPc= Kc # TPc= true_pos_class_c
                    Lc= np.sum(np.any((doa_error > self.doaT, dist_error>self.distT, reldist_error>self.reldistT, 
                            np.logical_and(np.logical_not(onscreen_correct), self.req_onscreen)), axis=0))
                    FPct= Lc # FPct= false_pos_class_c_threshold
                    FPc= FPcinf + FPct # FPc= False positive class c
                    TPct= Kc - FPct # TPct = True positive class c threshold
                    assert Pc == TPct+FPc
                    assert Rc == TPct + FPct + FNc

                    # update class sensitive information
                    self.total_doa_error[class_count]+= doa_error.sum()
                    self.total_dist_error[class_count]+= dist_error.sum()
                    self.total_reldist_error[class_count]+= reldist_error.sum()
                    self.total_onscreen_correct[class_count]+= onscreen_correct.sum()

                    self.true_pos[class_count]+= TPct
                    self.de_true_pos[class_count]+= TPc

                    self.false_pos[class_count]+= FPcinf
                    self.de_false_pos[class_count]+= FPcinf
                    self.false_pos_spatial[class_count]+= FPct
                    loc_false_pos+= FPc

                    self.false_neg[class_count]+= FNc
                    self.de_false_neg[class_count]+= FNc
                    loc_false_neg+= FNc
                
                elif class_count in labels[frame_count] and class_count not in preds[frame_count]:
                    # False negative.
                    loc_false_neg+= num_gt_doas
                    self.false_neg[class_count]+= num_gt_doas
                    self.de_false_neg[class_count]+= num_gt_doas
                
                elif class_count not in labels[frame_count] and class_count in preds[frame_count]:
                    # False positive.
                    loc_false_pos+= num_pred_doas
                    self.false_pos[class_count]+= num_pred_doas
                    self.de_false_pos[class_count]+= num_pred_doas
                else:
                    # True negative.
                    pass

class ComputeSeldResults:
    def __init__(self, dataset_cfg, metrics_cfg, num_classes=13):
        self.dataset_cfg= dataset_cfg
        self.metrics_cfg= metrics_cfg
        self.doaT= self.metrics_cfg['doa_threshold']
        self.distT= np.inf if self.metrics_cfg['dist_threshold']=='inf' else 0
        self.reldistT= self.metrics_cfg['reldist_threshold']
        self.req_onscreen= self.metrics_cfg['req_onscreen']

        # collect the reference files.
        self.ref_labels= {}
        self.desc_dir= os.path.join(self.dataset_cfg['root_dir'], 'metadata_dev')
        
        for split in os.listdir(self.desc_dir):
            split_folder= os.path.join(self.desc_dir, split)

            NUM_PROCESS= 15
            global run_task3
            def run_task3(ref_file):
                labels_dict, _= utils.load_labels(label_file= os.path.join(split_folder, ref_file), convert_to_cartesian= False)
                num_ref_frames= max(list(labels_dict.keys())) if len(labels_dict) > 0 else 0
                out= {ref_file: [utils.organize_labels(labels_dict= labels_dict, max_frames= num_ref_frames), num_ref_frames]}
                return out
            with Pool(NUM_PROCESS) as p:
                pool_results= p.map(run_task3, os.listdir(split_folder))
            print(f'Finished {split}')

            for i in range(len(pool_results)):
                ref_file = list(pool_results[i].keys())[0]
                self.ref_labels[ref_file]= pool_results[i][ref_file]
        
        self.num_ref_files= len(self.ref_labels)
        self.average= self.metrics_cfg['average']
        self.num_classes= num_classes

    def get_seld_results(self, pred_files_path):
        # collect the pred files info.
        pred_files= os.listdir(pred_files_path)
        seld_metrics= SeldMetrics(metrics_cfg= self.metrics_cfg, num_classes= self.num_classes)

        preds_labels_dict= {}

        for pred_count, pred_file in enumerate(pred_files):
            pred_file_path= os.path.join(pred_files_path, pred_file)
            preds_dict, _= utils.load_labels(label_file= pred_file_path, convert_to_cartesian= False)
            num_pred_frames= max(list(preds_dict.keys())) if len(preds_dict) > 0 else 0
            num_ref_frames= self.ref_labels[pred_file][1]
            pred_labels= utils.organize_labels(labels_dict= preds_dict, max_frames= max(num_pred_frames, num_ref_frames))
            # calculate scores.
            seld_metrics.update_seld_scores(preds= pred_labels, labels= self.ref_labels[pred_file][0])

            if self.metrics_cfg['use_jackknife']:
                preds_labels_dict[pred_file]= pred_labels

        # Overall SED and DOA scores.
        mean_results, class_wise_results= seld_metrics.compute_seld_scores()

        return mean_results, class_wise_results

if __name__=='__main__':
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    config_file = './config.yml'
    cfg= utils.get_configurations(config_file)
    dataset_cfg, full_model_cfg, training_cfg, metrics_cfg = cfg['DATASET'], cfg['MODEL'], cfg['TRAINING'], cfg['METRICS']

    seld_results= ComputeSeldResults(dataset_cfg= dataset_cfg, metrics_cfg= metrics_cfg, num_classes= 13)

