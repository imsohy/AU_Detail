''' training script of DECA
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy
from decalib.gatfarec_Video_OnlyExpress_WT_detail import DECA
from decalib.trainer_Video_OnlyExpress_WT_detail import Trainer

# temporal check if it runs in CPU
#cudnn.enabled = False
#cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True) #for NaN search
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))


def main(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
    # os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
    with open(os.path.join(cfg.output_dir, cfg.train.log_dir, 'full_config.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    shutil.copy(cfg.cfg_file, os.path.join(cfg.output_dir, 'config.yaml'))

    # cudnn related setting
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # start training
    # deca model
    # from decalib.mymodel import mymodel
    if 'detail' in cfg.cfg_file:
        cfg.rasterizer_type = 'pytorch3d'
        cfg.device = 'cuda' # determines what gpu we use in CUDA_VISIBLE_DEVICES. default cuda = cuda:0

        mymodel = DECA(cfg, device=cfg.device)
        trainer = Trainer(model=mymodel, config=cfg, device=cfg.device)

    else:
        print("no detail option")
    ## start train
    trainer.fit()

# increase weighted landmarks loss of mouth
if __name__ == '__main__':
    from decalib.utils.config_wt import parse_args
    cfg = parse_args(cfg_name='configs/release_version/deca_pretrain_video_OnlyE2_WT_detail.yml')
    
    if cfg.cfg_file is not None: 
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0] 
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml
