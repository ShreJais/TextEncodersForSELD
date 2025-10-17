import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools, os, time
from tqdm import tqdm

import pandas as pd
import utils, metrics

def optimizer_fn(cfg, params):
    if cfg['optimizer']=='adam':
        optimizer= torch.optim.Adam(params= params, lr= cfg['optim_conf']['adam']['lr'], 
            betas= cfg['optim_conf']['adam']['betas'],weight_decay= cfg['optim_conf']['adam']['weight_decay'])
    elif cfg['optimizer']=='adamw':
        optimizer= torch.optim.AdamW(params= params, lr= cfg['optim_conf']['adamw']['lr'], 
            betas= cfg['optim_conf']['adamw']['betas'], weight_decay= cfg['optim_conf']['adamw']['weight_decay'])
    return optimizer

def get_lrscheduler(cfg, optimizer, trainloader_len):
    if cfg['scheduler']=='cosine':
        lr_scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer= optimizer, T_max= cfg['max_epochs']*trainloader_len,
            eta_min= cfg['scheduler_conf']['cosine']['eta_ratio']*optimizer.param_groups[0]['lr'], verbose= True)
    else:
        lr_scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer= optimizer, milestones= [cfg['max_epochs']//4], gamma= 0.1, verbose= True)
    return lr_scheduler

# Seld loss.
class SELDLossADPIT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_cfg= cfg['MODEL']
        self.dataset_cfg= cfg['DATASET']

        self.mse_loss= nn.MSELoss(reduction= 'none')
        self.bce_loss= nn.BCEWithLogitsLoss(reduction= 'none')  
    
    def make_labels_permutations(self, x):
        label_perms= []
        perms= {0: [range(0, 1),  list(itertools.product([0], repeat=3)), 1], 
                1: [range(1, 7), list(itertools.product([1, 2], repeat=3)), 2], 
                2: [range(7, 13), list(itertools.product([3, 4, 5], repeat=3)), 3]}
        
        for key, values in perms.items():
            vals1, vals2, vals3= values
            vals2= [v for v in vals2 if len(set(v))==vals3]
            for val1, val2 in zip(vals1, vals2):
                # print(key, val1, val2)
                label_perms.append(torch.cat((x[:, :, val2[0], :, :], x[:, :, val2[1], :, :], x[:, :, val2[2], :, :]), dim=2))
        pads = {0: label_perms[1]+ label_perms[7], 1: label_perms[0]+label_perms[7], 2: label_perms[0]+label_perms[1]}
        for key, values in perms.items():
            vals1, _, _= values
            for val1 in vals1:
                # print(val1, key)
                label_perms[val1]= label_perms[val1]+pads[key]
        return label_perms
    
    def forward(self, preds, labels):
        '''
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 possible permutations.
        Args:
            preds:
                audio_visual: (bs, 1, 156) -> 156 = 3(tracks) * 4 (act*x, act*y, dist, on/off) * 13(classes)
            labels:
                audio_visual: (bs, 1, 6, 5, 13) -> 6(dummy tracks), 5(act, x, y, dist, on/off), 13 (classes)
        '''
        if labels.ndim==4:
            labels= labels.unsqueeze(1) # (batch_size, 1, 6, 5, 13)
        if labels.shape[-2]==5:
            labels= labels[:, :, :, :4, :] # (batch_size, 1, 6, 4, 13)

        batch_size, n_frames= labels.shape[:2]
        n_tracks, n_ele= preds.shape[2:4]
        n_classes, n_permutations= labels.shape[-1], 13

        preds_accdoa= preds.view(batch_size, n_frames, n_tracks, n_ele, n_classes)[:, :, :, 0:3, :]
        preds_accdoa= preds_accdoa.reshape(batch_size, n_frames, -1, n_classes)
        # calculate ACCDOA loss.
        labels_accdoa= torch.zeros(size= (batch_size, n_frames, 6, 3, n_classes)).to(preds.device)
        
        # i=0 --> No overlap from the same class.
        # i=1,2 --> Overlapping with 2 sources from the same class.
        # i=3,4,5 --> Overlapping with 3 sources from the same class.
        # for loop: this loop is creating the labels for the ACCDOA loss 
        #           by multiplying the relevant parts of the input labels tensor.

        for i in range(labels.shape[2]):
            labels_accdoa[:, :, i, :, :]= labels[:, :, i, 0:1, :]*labels[:, :, i, 1:4, :]

        labels_accdoa_perms_list= self.make_labels_permutations(x= labels_accdoa)

        accdoa_loss= []
        for label_perms in labels_accdoa_perms_list:
            accdoa_loss.append(self.mse_loss(preds_accdoa, label_perms).mean(dim= 2))
        
        if self.dataset_cfg['modality']=='audio':
            screen_loss= [0,] * n_permutations
        else:
            preds_screen= preds.view(batch_size, n_frames, n_tracks, n_ele, n_classes)[:, :, :, 3:4, :]
            preds_screen= preds_screen.reshape(batch_size, n_frames, -1, n_classes)
            labels_screen= torch.zeros(size= (batch_size, n_frames, 6, 1, n_classes)).to(preds.device)

            for i in range(labels.shape[2]):
                labels_screen[:, :, i, :, :]= labels[:, :, i, 4:5, :]
        
            labels_screen_perms_list= self.make_labels_permutations(x= labels_screen)
            screen_loss= []
        
            for label_perms in labels_screen_perms_list:
                screen_loss.append(self.bce_loss(preds_screen.float(), label_perms.float()).mean(dim= 2))

        # choose the permutation with the lowest ACCDOA loss.
        min_accdoa_loss_idx= torch.min(torch.stack(accdoa_loss, dim= 0), dim= 0).indices
        total_loss= 0
        for i in range(n_permutations):
            total_loss+= (accdoa_loss[i]+screen_loss[i])*(min_accdoa_loss_idx==i)
        return total_loss.mean()
    
def training(cfg, model: nn.Module, dir, summary_writer, train_loader, test_loader, device, best_fscore, start_epoch= 0):
    training_cfg= cfg['TRAINING']
    dataset_cfg= cfg['DATASET']
    metrics_cfg= cfg['METRICS']
    
    # output loss function.
    out_loss_fn= SELDLossADPIT(cfg= cfg)
    # seld metrics.
    seld_metrics= metrics.ComputeSeldResults(dataset_cfg= dataset_cfg, metrics_cfg= metrics_cfg, num_classes= dataset_cfg['num_classes'])
    # seld_metrics= None

    # optimizer and lr scheduler.
    optimizer= optimizer_fn(cfg= training_cfg, params= model.parameters())
    lr_scheduler= get_lrscheduler(cfg= training_cfg, optimizer= optimizer, trainloader_len= len(train_loader))

    if training_cfg['restore_from_checkpoints']:
        print("Loading model weights and optimizer state dict from initial checkpoint.")
        model_ckpt= torch.load(os.path.join(training_cfg['initial_checkpoints'], 'best_model.pth'), map_location= device, weights_only= False)
        model.load_state_dict(model_ckpt['seld_model'])
        optimizer.load_state_dict(model_ckpt['opt'])
        start_epoch= model_ckpt['epoch']+1
        best_fscore= model_ckpt['best_fscore']
        best_fscore_epoch= model_ckpt['best_fscore_epoch']
        best_epoch= model_ckpt['best_epoch']
        best_step= model_ckpt['best_step']

    loss_weights= training_cfg['loss_weights']
    
    # history.
    train_hist= pd.DataFrame(columns= ['epoch', 'tloss', 'seld_loss'])
    test_hist= pd.DataFrame(columns= ['epoch', 'tloss', 'seld_loss', 'fscore', 'doa_error', 'dist_error', 'reldist_error'])

    trainstep_hist= pd.DataFrame(columns= ['step', 'tloss', 'seld_loss'])
    teststep_hist= pd.DataFrame(columns= ['step', 'tloss', 'seld_loss'])

    print(f'Starting training for {training_cfg["max_epochs"]} epochs.')
    print(f'Using device: {device}')
    print(f'Initial best fscore: {best_fscore}')
    print(f'Loss weights: {loss_weights}')

    model.to(device)
    print(f'Check if model is in cuda: {next(model.parameters()).is_cuda}')

    start_time= time.time()
    val_step= 0
    best_epoch= -1 if start_epoch==0 else best_epoch
    best_fscore_epoch= -1 if start_epoch==0 else best_fscore_epoch
    best_step= -1 if start_epoch==0 else best_step

    for epoch in range(start_epoch, training_cfg['max_epochs']):
        epoch_st= time.time()
        trainepoch_hist, trainstep_hist, teststep_hist, val_step= train_epoch(cfg= cfg, epoch= epoch, model= model, train_loader= train_loader, 
            test_loader= test_loader, optimizer= optimizer, lr_scheduler= lr_scheduler, out_loss_fn= out_loss_fn, device= device, val_step= val_step, 
            teststep_hist= teststep_hist, trainstep_hist= trainstep_hist, best_step= best_step, dir= dir, 
            loss_weights= loss_weights)

        train_hist= pd.concat([train_hist, pd.DataFrame([{'epoch': epoch, 'tloss': trainepoch_hist['tloss'],'seld_loss': trainepoch_hist['seld_loss']}])], ignore_index= True)
        
        valepoch_hist, mean_results, class_wise_results= validation_epoch(cfg= cfg, epoch= epoch, model= model, val_loader= test_loader, out_loss_fn= out_loss_fn, metric= seld_metrics, device= device, 
            out_dir= dir['output_dir'], mode= 'random', loss_weights= loss_weights)
        
        test_hist= pd.concat([test_hist, 
            pd.DataFrame([{'epoch': epoch, 'tloss': valepoch_hist['tloss'], 'seld_loss': valepoch_hist['seld_loss'], 
            'fscore': valepoch_hist['fscore'], 'doa_error': valepoch_hist['doa_error'], 'dist_error': valepoch_hist['dist_error'], 
            'reldist_error': valepoch_hist['reldist_error']}])], ignore_index= True)
        
        epoch_et= time.time()

        print(f"Epoch: {epoch+1}/{training_cfg['max_epochs']} | Train -- tloss: {trainepoch_hist['tloss']:.4f} | "
              f"SeldLoss: {trainepoch_hist['seld_loss']:.4f}")
        print(f"Epoch: {epoch+1}/{training_cfg['max_epochs']} | Validation -- tloss: {valepoch_hist['tloss']:.4f} |"
              f"SeldLoss: {valepoch_hist['seld_loss']:.4f} | ", 
              f"Fscore: {valepoch_hist['fscore']:.4f} | DoaError: {valepoch_hist['doa_error']:.4f} | DistError: {valepoch_hist['dist_error']:.4f} | "
              f"RelDistError: {valepoch_hist['reldist_error']:.4f} | ")
        print(f'time taken for epoch: {(epoch_et-epoch_st)/60:.2f}min')
        
        # model parameters saved based on best fscore.
        if len(test_hist)==1 or valepoch_hist['fscore'] > best_fscore:
            best_fscore= valepoch_hist['fscore']
            best_fscore_epoch= epoch
            model_name= cfg['MODEL']['model_name']
            model_path= os.path.join(dir['checkpoint_dir'], model_name, 'best_fscore_model')
            os.makedirs(model_path, exist_ok= True)
            save_dict= {'seld_model': model.state_dict(), 'cfg': cfg, 'opt': optimizer.state_dict(), 
                'epoch': epoch, 'best_fscore': best_fscore, 'best_fscore_epoch': best_fscore_epoch, 
                'best_epoch': epoch, 'best_step': val_step}
            torch.save(save_dict, os.path.join(model_path, 'best_model.pt'))

        # model parameters saved based on minimum loss.
        if len(test_hist)==1 or valepoch_hist['tloss']< test_hist['tloss'].iloc[best_epoch]:
            best_epoch= epoch
            model_name= cfg['MODEL']['model_name']
            model_path= os.path.join(dir['checkpoint_dir'], model_name, 'best_loss_model')
            os.makedirs(model_path, exist_ok= True)
            save_dict= {'seld_model': model.state_dict(), 'cfg': cfg, 'opt': optimizer.state_dict(), 
                'epoch': epoch, 'best_fscore': valepoch_hist['fscore'], 'best_fscore_epoch': epoch, 
                'best_epoch': epoch, 'best_step': val_step}
            torch.save(save_dict, os.path.join(model_path, 'best_model.pt'))

        print_classwise_results(cwr= class_wise_results, metrics_cfg= metrics_cfg, dataset_cfg= dataset_cfg)

        train_hist.to_csv(os.path.join(dir['checkpoint_dir'], 'train_hist.csv'))
        test_hist.to_csv(os.path.join(dir['checkpoint_dir'], 'test_hist.csv'))
        trainstep_hist.to_csv(os.path.join(dir['checkpoint_dir'], 'trainstep_hist.csv'))
        teststep_hist.to_csv(os.path.join(dir['checkpoint_dir'], 'teststep_hist.csv'))

    end_time= time.time()
    print(f'Total time taken: {(end_time-start_time)/60:.2f} minutes.')
    
    # Test the model based on the best fscore.
    best_fscore_model_ckpt= torch.load(os.path.join(dir['checkpoint_dir'], model_name, 'best_fscore_model', 'best_model.pt'), weights_only= False)
    model.load_state_dict(best_fscore_model_ckpt['seld_model'])
    valepoch_hist, mean_results, class_wise_results= validation_epoch(cfg= cfg, epoch= 0, model= model, 
        val_loader= test_loader, out_loss_fn= out_loss_fn, metric= seld_metrics, device= device, 
        out_dir= dir['output_dir'], mode= 'random', loss_weights= loss_weights)
    
    print(f'Test Results on the best fscore model.')
    print(f" Test -- tloss: {valepoch_hist['tloss']:.4f} |"
          f"SeldLoss: {valepoch_hist['seld_loss']:.4f} | "
          f"Fscore: {valepoch_hist['fscore']:.4f} | DoaError: {valepoch_hist['doa_error']:.4f} | DistError: {valepoch_hist['dist_error']:.4f} | "
          f"RelDistError: {valepoch_hist['reldist_error']:.4f}")
    print_classwise_results(cwr= class_wise_results, metrics_cfg= metrics_cfg, dataset_cfg= dataset_cfg)

    print('=============================================')
    best_loss_model_ckpt= torch.load(os.path.join(dir['checkpoint_dir'], model_name, 'best_loss_model', 'best_model.pt'), weights_only= False)
    model.load_state_dict(best_loss_model_ckpt['seld_model'])
    valepoch_hist, mean_results, class_wise_results= validation_epoch(cfg= cfg, epoch= 0, model= model, 
        val_loader= test_loader, out_loss_fn= out_loss_fn, metric= seld_metrics, device= device, 
        out_dir= dir['output_dir'], mode= 'random', loss_weights= loss_weights)
    
    print(f'Test Results on the best loss model.')
    print(f" Test -- tloss: {valepoch_hist['tloss']:.4f} |"
          f"SeldLoss: {valepoch_hist['seld_loss']:.4f} | "
          f"Fscore: {valepoch_hist['fscore']:.4f} | DoaError: {valepoch_hist['doa_error']:.4f} | DistError: {valepoch_hist['dist_error']:.4f} | "
          f"RelDistError: {valepoch_hist['reldist_error']:.4f}")
    print_classwise_results(cwr= class_wise_results, metrics_cfg= metrics_cfg, dataset_cfg= dataset_cfg)


def train_epoch(cfg, epoch, model, train_loader, test_loader, optimizer, lr_scheduler, out_loss_fn, device, trainstep_hist, teststep_hist, 
    dir, loss_weights, mode= 'random', val_step= 0, best_step= 0):
    model.train()
    epoch_hist= {'tloss': 0, 'seld_loss': 0}
    tstep_hist= {'tloss': 0, 'seld_loss': 0}
    # val_iter= iter(test_loader)

    t= tqdm(iterable= train_loader, leave= False)
    num_steps= 0

    for batch in t:
        event_specs, audio_iv_specs, dist_specs, target_labels, orig_labels, file_names = batch
        event_specs, audio_iv_specs= event_specs.to(device), audio_iv_specs.to(device)
        dist_specs, target_labels, orig_labels= dist_specs.to(device), target_labels.to(device), orig_labels.to(device)

        batch_size= event_specs.shape[0]
        optimizer.zero_grad()

        seld_preds= model(audio_specs= event_specs, audio_ivspecs= audio_iv_specs, dist_specs= dist_specs)
        seld_loss= out_loss_fn(preds= seld_preds, labels= target_labels)
        total_loss= loss_weights[1]*seld_loss

        total_loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        num_steps+= 1
        # Update the epoch history.
        epoch_hist= update_history(hist= epoch_hist, total_loss= total_loss, seld_loss= seld_loss)
        tstep_hist= update_history(hist= tstep_hist, total_loss= total_loss, seld_loss= seld_loss)
        t.set_description(
            f'Epoch: {epoch+1}, Loss: {epoch_hist["tloss"]/num_steps:.4f}, '  
            f'SELD Loss: {epoch_hist["seld_loss"]/num_steps:.4f}')

        # if num_steps%1000==0:
        #     val_step+= 1
        #     trainstep_hist= trainstep_hist.append({'step': val_step, 'tloss': tstep_hist['tloss']/1000, 'embedding_loss': tstep_hist['embedding_loss']/1000, 
        #         'seld_loss': tstep_hist['seld_loss']/1000}, ignore_index= True)
        #     tstep_hist= {'tloss': 0, 'embedding_loss': 0, 'seld_loss': 0}

        #     step_hist, val_iter= validate_steps(model= model, val_loader= test_loader, out_loss_fn= out_loss_fn, device= device, 
        #         mode= mode, num_batches= 800, val_iter= val_iter, loss_weights= loss_weights)

        #     teststep_hist= teststep_hist.append({'step': val_step, 'tloss': step_hist['tloss'], 'embedding_loss': step_hist['embedding_loss'], 
        #         'seld_loss': step_hist['seld_loss']}, ignore_index= True)
            
        #     if len(teststep_hist)==1 or step_hist['tloss']< teststep_hist['tloss'].iloc[best_step]:
        #         best_step= val_step
        #         model_name= cfg['MODEL']['model_name']
        #         model_path= os.path.join(dir['checkpoint_dir'], model_name, 'best_step_model')
        #         os.makedirs(model_path, exist_ok= True)
        #         save_dict= {'seld_model': model.state_dict(), 'cfg': cfg, 'opt': optimizer.state_dict(), 
        #             'epoch': epoch, 'best_epoch': epoch, 'best_step': val_step}
        #         torch.save(save_dict, os.path.join(model_path, 'best_model.pt'))
        
    for key in epoch_hist.keys():
        epoch_hist[key]/= num_steps
    return epoch_hist, trainstep_hist, teststep_hist, val_step

# def validate_steps(model, val_loader, out_loss_fn, device, loss_weights, mode= 'random', num_batches= 800, val_iter= None):
#     model.eval()
#     step_hist= {'tloss': 0, 'embedding_loss': 0, 'seld_loss': 0}
#     if val_iter is None:
#         val_iter= iter(val_loader)

#     t= tqdm(iterable= itertools.islice(val_iter, num_batches), leave= False, total= num_batches)
#     num_steps= 0
#     with torch.no_grad():
#         for batch in t:
#             event_specs, audio_iv_specs, ipd_ild_specs, dist_specs, target_labels, orig_labels, file_names = batch
#             event_specs, audio_iv_specs, ipd_ild_specs= event_specs.to(device), audio_iv_specs.to(device), ipd_ild_specs.to(device)
#             dist_specs, target_labels, orig_labels= dist_specs.to(device), target_labels.to(device), orig_labels.to(device)
            
#             batch_size= event_specs.shape[0]

#             prompt_templates, _, _= utils.get_prompt_template(mode= mode, num_template= batch_size)
#             encoded_tokens, pos= model.get_placeholder_tokens(prompt_texts= prompt_templates)
#             encoded_tokens, pos= encoded_tokens.to(device), torch.as_tensor(pos).to(device)

#             final_predembeddings, seld_preds= model(audio_specs= event_specs, audio_ivspecs= audio_iv_specs, 
#                 ipd_ildspecs= ipd_ild_specs, dist_specs= dist_specs, encoded_tokens= encoded_tokens, pos= pos)
#             final_gtembeddings= model.get_original_label_embeddings(orig_labels= orig_labels, encoded_tokens= encoded_tokens, pos= pos)

#             # compute loss.
#             embed_loss= embedding_loss(pred_emb= final_predembeddings, target_emb= final_gtembeddings, beta= 1/0.07)
#             seld_loss= out_loss_fn(preds= seld_preds, labels= target_labels)
#             total_loss= loss_weights[0]*embed_loss+loss_weights[1]*seld_loss

#             # update the history.
#             num_steps+= 1
#             step_hist= update_history(hist= step_hist, total_loss= total_loss, embed_loss= embed_loss, seld_loss= seld_loss)
#             t.set_description(
#                 f'Loss: {step_hist["tloss"]/num_steps:.4f}, ' 
# 			    f'EmbedLoss: {step_hist["embedding_loss"]/num_steps:.4f}, ' 
#                 f'SELD Loss: {step_hist["seld_loss"]/num_steps:.4f}')  
            
#     for key in step_hist.keys():
#         step_hist[key]/= num_batches
#     return step_hist, val_iter

def validation_epoch(cfg, epoch, model, val_loader, out_loss_fn, metric, device, out_dir, loss_weights, mode= 'random'):
    model.eval()
    epoch_hist= {'tloss': 0, 'seld_loss': 0, 'fscore': 0, 'doa_error': 0, 'dist_error': 0, 'reldist_error': 0}
    num_steps= 0
    t= tqdm(iterable= val_loader, leave= False)
    
    with torch.no_grad():
        for batch in t:
            event_specs, audio_iv_specs, dist_specs, target_labels, orig_labels, file_names = batch
            event_specs, audio_iv_specs= event_specs.to(device), audio_iv_specs.to(device)
            dist_specs, target_labels, orig_labels= dist_specs.to(device), target_labels.to(device), orig_labels.to(device)

            batch_size= event_specs.shape[0]

            seld_preds= model(audio_specs= event_specs, audio_ivspecs= audio_iv_specs, dist_specs= dist_specs)
            seld_loss= out_loss_fn(preds= seld_preds, labels= target_labels)
            total_loss= loss_weights[1]*seld_loss

            # save predictions to csv files for metric calculations.
            utils.write_preds_to_dcase_format(preds= seld_preds, dataset_cfg= cfg['DATASET'], metrics_cfg= cfg['METRICS'], 
                out_dir= out_dir, file_list= file_names, split= 'dev-test')
            
            # update the history.
            num_steps+= 1
            epoch_hist= update_history(hist= epoch_hist, total_loss= total_loss, seld_loss= seld_loss)
            t.set_description(
                f'Epoch: {epoch+1}, Loss: {epoch_hist["tloss"]/num_steps:.4f}, ' 
                f'SELD Loss: {epoch_hist["seld_loss"]/num_steps:.4f}')

    for key in epoch_hist.keys():
        epoch_hist[key]/= len(val_loader)

    mean_results, class_wise_results= metric.get_seld_results(pred_files_path= os.path.join(out_dir, 'dev-test'))
    
    epoch_hist['fscore']= mean_results['fscore']
    epoch_hist['doa_error']= mean_results['doa_error']
    epoch_hist['dist_error']= mean_results['dist_error']
    epoch_hist['reldist_error']= mean_results['reldist_error']
    return epoch_hist, mean_results, class_wise_results

def update_history(hist, total_loss, seld_loss):
	hist['tloss']+= total_loss.item()
	hist['seld_loss']+= seld_loss.item()
	return hist

def print_classwise_results(cwr, metrics_cfg, dataset_cfg):
	if metrics_cfg['average']=='macro':
		print('Class-wise Results')
		
		if dataset_cfg['modality']=='audio_visual':
			print('Class\tFS\tDoaE\tDistE\tRelDistE\tOnscreenAcc.')
		else:
			print('Class\tFS\tDoaE\tDistE\tRelDistE')

		for ccnt in range(dataset_cfg['num_classes']):
			if dataset_cfg['modality']=='audio_visual':
				print(f"{ccnt:0.2f}\t{cwr[0][ccnt]:0.2f}\t{cwr[1][ccnt]:0.2f}\t{cwr[2][ccnt]:0.2f}\t{cwr[3][ccnt]:0.2f}\t{cwr[4][ccnt]:0.2f}")
			else:
				print(f"{ccnt:0.2f}\t{cwr[0][ccnt]:0.2f}\t{cwr[1][ccnt]:0.2f}\t{cwr[2][ccnt]:0.2f}\t{cwr[3][ccnt]:0.2f}")