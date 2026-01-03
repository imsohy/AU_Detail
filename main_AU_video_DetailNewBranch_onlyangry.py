'''
디테일뉴(기본 )->디테일뉴브랜치(latent code 제거)->디코더 트레인
->v3 (코얼스 제거)-> onlyangry 2025 1228
'''
import os, sys
import numpy as np
import yaml
import torch
import torch.backends.cudnn as cudnn
import torch
import shutil
from copy import deepcopy
from decalib.gatfarec_Video_DetailNewBranch_v3 import DECA
from decalib.trainer_Video_DetailNewBranch_onlyangry import Trainer
from decalib.utils.config_wt_DetailNewBranch_v3 import parse_args

# temporal check if it runs in CPU
#cudnn.enabled = False
#cudnn.benchmark = False
#torch.autograd.set_detect_anomaly(True) #for NaN search <---this may cause slow down
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

    cfg.rasterizer_type = 'pytorch3d'

    use_multi = getattr(cfg, "multi_gpu",False)    #check whether we use multi gpu
    base_device = getattr(cfg, "device", "cuda:0")
    device_detail = getattr(cfg, "device_detail", "cuda:1")

    # start training
    # deca model

    if use_multi and device_detail is not None:
        mymodel = DECA(
            cfg,
            device=base_device,
            device_detail=getattr(cfg, "device_detail", base_device),
            multi_gpu=True
        )
        print(f"Using multi GPU")
    else:
        mymodel = DECA(
            cfg,
            device=base_device,
            device_detail=base_device,
            multi_gpu=False
        )
        print(f"Using single GPU")
    trainer = Trainer(model=mymodel, config=cfg, device=cfg.device)
    ## start train
    trainer.fit()

# increase weighted landmarks loss of mouth
if __name__ == '__main__':
    cfg = parse_args(cfg_name='configs/release_version/deca_pretrain_video_OnlyE2_DetailNewBranch_decodertrain_v3.yml')
    
    # Check if at least one training option is enabled
    if not cfg.train.train_coarse and not cfg.train.train_detail:
        print("[config] please turn at least one true; train_coarse or train_detail")
        exit(1)
    
    if cfg.cfg_file is not None: 
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0] 
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_train.py --cfg configs/release_version/deca_pretrain.yml
