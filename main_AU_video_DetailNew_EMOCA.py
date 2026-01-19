"""
2026 01 19
EMOCA 기반 Detail 학습용 main 파일
이미 학습된 Coarse 모델을 사용하여 Detail만 학습합니다.
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
from decalib.gatfarec_Video_OnlyExpress import DECA  # EMOCA 기반 통합 버전
from decalib.trainer_Video_OnlyExpress import Trainer
from decalib.utils.config import parse_args

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
    # 참고: AU_Detail_legacy/main_AU_video_DetailNew_20260104.py:61-67
    # multi_gpu 관련 로직 제거 (사용자 요청)
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
    cfg = parse_args(cfg_name='configs/release_version/deca_pretrain_video_OnlyE2_Detail.yml')
    
    # Check if at least one training option is enabled
    if not cfg.train.train_coarse and not cfg.train.train_detail:
        print("[config] please turn at least one true; train_coarse or train_detail")
        exit(1)
    
    # Check if pretrained_coarse_modelpath is set when train_coarse=False
    if not cfg.train.train_coarse and not hasattr(cfg, 'pretrained_coarse_modelpath'):
        print("[config] ERROR: train_coarse=False requires pretrained_coarse_modelpath to be set!")
        exit(1)
    
    if cfg.cfg_file is not None: 
        exp_name = cfg.cfg_file.split('/')[-1].split('.')[0] 
        cfg.exp_name = exp_name
    main(cfg)

# run:
# python main_AU_video_DetailNew_EMOCA.py

