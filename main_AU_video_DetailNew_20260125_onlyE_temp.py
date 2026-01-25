"""
2026 01 25
copied from: main_AU_video_DetailNew_20260104.py
OnlyExpressionA 방식: Coarse ViT가 expression만 추론하고, tex와 light는 DECA에서 가져옴.
Multi-GPU 설정 제거하여 단일 GPU로 단순화.
"""
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
from decalib.gatfarec_Video_DetailNew_20260125_onlyE_temp import DECA #20260125_onlyE_temp (DEBUG VERSION)
from decalib.trainer_Video_DetailNew_20260125_onlyE_temp import Trainer
from decalib.utils.config_DetailNew_20260125_onlyE import parse_args

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

    # start training
    # deca model
    # Multi-GPU 관련 로직 제거 (사용자 요청)
    base_device = getattr(cfg, "device", "cuda:0")
    
    mymodel = DECA(
        config=cfg,
        device=base_device
    )
    print(f"Using device: {base_device}")
    
    trainer = Trainer(model=mymodel, config=cfg, device=base_device)
    ## start train
    trainer.fit()

# increase weighted landmarks loss of mouth
if __name__ == '__main__':
    # Config 파일 선택 (OnlyE 또는 OnlyE2)
    # cfg = parse_args(cfg_name='configs/release_version/deca_pretrain_video_20260125_onlyE.yml')  # AU Loss 있음
    cfg = parse_args(cfg_name='configs/release_version/deca_pretrain_video_20260125_onlyE2.yml')  # AU Loss 없음
    
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
