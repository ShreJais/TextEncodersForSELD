import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchinfo import summary

from pretraining import utils, dataset
import models, train_utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(cfg, device):
    # Get specific configuration.
    dataset_cfg, model_cfg, training_cfg, metrics_cfg = cfg['DATASET'], cfg['MODEL'], cfg['TRAINING'], cfg['METRICS']
    dir, summary_writer= utils.setup(cfg= cfg, dataset_cfg= dataset_cfg, model_cfg= model_cfg, training_cfg= training_cfg)

    # dataloader.
    train_loader= dataset.get_dataloader(cfg= cfg, mode= 'dev_train')
    test_loader= dataset.get_dataloader(cfg= cfg, mode= 'dev_test')

    print(f'len(train_loader): {len(train_loader)}')
    print(f'len(test_loader): {len(test_loader)}')

    print('Model configurations')
    print(f'Model name: {model_cfg["model_name"]}')
    print(f'audio backbone: {model_cfg["audio_backbone"]}, doa backbone: {model_cfg["doa_backbone"]}, distance backbone: {model_cfg["distance_backbone"]}')
    print(f'audio proj: {model_cfg["audio_proj"]}, doa proj: {model_cfg["doa_proj"]}, distance proj: {model_cfg["distance_proj"]}')
    print(f'decoder: {model_cfg["decoder"]}, seld out: {model_cfg["seld_out"]}')
    print(f'audio normalization: {model_cfg["audio_normalization"]}')
    print(f'text encoder: {model_cfg["text_encoder"]}')
    print(f'num_feats: {model_cfg["num_feats"]}, embed_dim: {model_cfg["embed_dim"]}, case: {model_cfg["case"]}')
    print()

    pretrained_config_file= model_cfg['pretrain_cfg']
    pretrained_cfg= utils.get_configurations(config_file= pretrained_config_file)
    pretrained_model_cfg= pretrained_cfg['MODEL']

    basemodel= models.SELDNet(pretrained_model_cfg= pretrained_model_cfg, model_cfg= model_cfg)
    batch_size= training_cfg['train_batch_size']
    backbone_inch= pretrained_model_cfg['backbone_inch'][pretrained_model_cfg['case']]
    
    prompt_templates, _, _= utils.get_prompt_template(mode= 'random', num_template= batch_size, num_frames= 50)
    encoded_tokens, pos= basemodel.get_placeholder_tokens(prompt_texts= prompt_templates)
    
    input_data= [torch.randn(batch_size, backbone_inch['audio'], 251, 64).to(device), 
                 torch.randn(batch_size, backbone_inch['audio_iv'], 251, 64).to(device), 
                 torch.randn(batch_size, backbone_inch['dist'], 251, 64).to(device), 
                 encoded_tokens.to(device), torch.as_tensor(pos).to(device)]

    summary(model= basemodel.to(device), input_data= input_data, depth= 2,  
        col_names= ['input_size', 'output_size', 'num_params', 'trainable', 'params_percent'])
        
    start_epoch= 0
    best_fscore= float('-inf')

    train_utils.training(cfg= cfg, model= basemodel, dir= dir, summary_writer= summary_writer, 
        train_loader= train_loader, test_loader= test_loader, device= device, best_fscore= best_fscore, start_epoch= start_epoch)

if __name__=='__main__':
    device= 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # set seed.
    utils.set_seed(seed=42)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    config_file= './config.yml'
    cfg= utils.get_configurations(config_file= config_file)

    main(cfg= cfg, device= device)