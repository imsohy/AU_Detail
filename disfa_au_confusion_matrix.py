# disfa_au_confusion_matrix.py
# -*- coding: utf-8 -*-
"""
DISFA 데이터셋을 사용하여 AU detection 모델의 confusion matrix를 생성합니다.
각 AU별로 confusion matrix를 생성하고 시각화합니다.
"""

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# sklearn import 제거 (numpy/pandas 버전 호환성 문제 해결)
# from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from decalib.gatfarec_Video_OnlyExpress_WT_DetailNew import DECA
from decalib.datasets import datasets2 as datasets
from decalib.utils.config_wt_DetailNew import cfg as deca_cfg
from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.conf import get_config, set_logger, set_outdir, set_env


class DISFADataset(Dataset):
    """DISFA 데이터셋 로더"""
    def __init__(self, disfa_root, sequences, au_idx=[1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]):
        """
        Args:
            disfa_root: DISFA 데이터셋 루트 경로 (예: '/path/to/DISFA')
            sequences: 사용할 시퀀스 리스트 (예: ['SN023', 'SN025', ...])
            au_idx: DISFA에서 사용하는 AU 인덱스 리스트
        """
        self.disfa_root = disfa_root
        self.sequences = sequences
        self.au_idx = au_idx
        self.num_aus = len(au_idx)
        
        # 이미지 경로와 레이블 로드
        self.image_paths = []
        self.labels = []
        
        label_path = os.path.join(disfa_root, 'ActionUnit_Labels')
        
        for seq in sequences:
            seq_path = os.path.join(label_path, seq)
            au1_path = os.path.join(seq_path, f'{seq}_au1.txt')
            
            if not os.path.exists(au1_path):
                print(f"Warning: {au1_path} not found, skipping {seq}")
                continue
            
            # 프레임 수 확인
            with open(au1_path, 'r') as f:
                total_frames = len(f.readlines())
            
            # AU 레이블 로드
            au_label_array = np.zeros((total_frames, self.num_aus), dtype=np.int32)
            
            for ai, au in enumerate(au_idx):
                au_label_path = os.path.join(seq_path, f'{seq}_au{au}.txt')
                if not os.path.exists(au_label_path):
                    continue
                
                with open(au_label_path, 'r') as f:
                    for t, line in enumerate(f.readlines()):
                        frame_idx, au_intensity = line.strip().split(',')
                        au_intensity = int(au_intensity)
                        au_label_array[t, ai] = 1 if au_intensity >= 1 else 0
            
            # 이미지 경로와 레이블 저장
            for frame_idx in range(total_frames):
                # 이미지 경로 (croppedImages2 폴더 사용)
                img_path = os.path.join(disfa_root, 'video', seq, 'croppedImages2', f'{frame_idx}.png')
                # 만약 croppedImages2가 없다면 다른 경로 시도
                if not os.path.exists(img_path):
                    img_path = os.path.join(disfa_root, 'video', seq, 'originalImages', f'{frame_idx}.png')
                
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(au_label_array[frame_idx])
        
        print(f"Loaded {len(self.image_paths)} images from {len(sequences)} sequences")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 이미지 로드 (datasets2.TestData와 동일한 방식 사용)
        # 실제로는 datasets2.TestData를 사용하는 것이 더 나을 수 있습니다
        return {
            'image_path': img_path,
            'label': torch.from_numpy(label).float(),
            'idx': idx
        }


def confusion_matrix_numpy(y_true, y_pred, labels=None):
    """
    numpy를 사용한 confusion matrix 구현
    sklearn.metrics.confusion_matrix의 대체 함수
    """
    if labels is None:
        labels = [0, 1]
    
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    # labels를 딕셔너리로 변환하여 인덱스 찾기 빠르게
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for i in range(len(y_true)):
        true_val = int(y_true[i])
        pred_val = int(y_pred[i])
        
        # labels에 없는 값은 무시
        if true_val in label_to_idx and pred_val in label_to_idx:
            true_idx = label_to_idx[true_val]
            pred_idx = label_to_idx[pred_val]
            cm[true_idx, pred_idx] += 1
    
    return cm


def create_au_confusion_matrix(model, dataloader, num_aus, device='cuda', threshold=0.5):
    """
    AU별 confusion matrix를 생성합니다.
    
    Args:
        model: AU detection 모델
        dataloader: 데이터 로더
        num_aus: AU 개수
        device: 디바이스
        threshold: 이진화 임계값
    
    Returns:
        confusion_matrices: 각 AU에 대한 confusion matrix 리스트
        all_preds: 모든 예측값
        all_labels: 모든 레이블
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # batch 구조에 맞게 수정 필요
            # 실제 모델 입력 형식에 맞게 조정해야 합니다
            images = batch['image'].to(device)  # [B, 3, H, W] 또는 [B, 3, 3, H, W]
            labels = batch['label'].to(device)  # [B, num_aus]
            
            # 모델 예측
            # 모델 출력 형식에 맞게 조정 필요
            outputs = model(images)  # MEFARG의 경우 [main_output, sub_output] 반환 가능
            
            # 출력 형식에 따라 처리
            if isinstance(outputs, tuple):
                outputs = outputs[1]  # sub_output 사용 (또는 main_output)
            
            preds = (torch.sigmoid(outputs) >= threshold).long()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # AU별 confusion matrix 생성
    confusion_matrices = []
    for au_idx in range(num_aus):
        cm = confusion_matrix_numpy(all_labels[:, au_idx], all_preds[:, au_idx], labels=[0, 1])
        confusion_matrices.append(cm)
    
    return confusion_matrices, all_preds, all_labels


def plot_au_confusion_matrix(cm_list, au_names, save_path=None, figsize=(16, 12)):
    """
    AU별 confusion matrix를 시각화합니다.
    """
    num_aus = len(cm_list)
    cols = 4
    rows = (num_aus + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if num_aus > 1 else [axes]
    
    for idx, (cm, au_name) in enumerate(zip(cm_list, au_names)):
        ax = axes[idx]
        
        # 정규화된 confusion matrix
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   vmin=0, vmax=1)
        ax.set_title(f'{au_name}\n(TP:{cm[1,1]}, FP:{cm[0,1]}, FN:{cm[1,0]}, TN:{cm[0,0]})')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # 빈 subplot 숨기기
    for idx in range(num_aus, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix plot to {save_path}")
    plt.show()


def print_metrics(cm_list, au_names):
    """각 AU별 성능 지표 출력"""
    print("\n" + "="*80)
    print("AU별 성능 지표")
    print("="*80)
    print(f"{'AU':<8} {'TP':<8} {'FP':<8} {'FN':<8} {'TN':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*80)
    
    for idx, (cm, au_name) in enumerate(zip(cm_list, au_names)):
        TP = cm[1, 1]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[0, 0]
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{au_name:<8} {TP:<8} {FP:<8} {FN:<8} {TN:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("="*80)


def main(args):
    device = args.device
    
    # DISFA 설정
    DISFA_AU_IDX = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
    DISFA_AU_NAMES = [f'AU{au}' for au in DISFA_AU_IDX]
    
    # DISFA 테스트 시퀀스
    TEST_DISFA_Sequence_split = ['SN023', 'SN025', 'SN008', 'SN005', 'SN007', 'SN017', 'SN004', 'SN001', 'SN026']
    
    # 모델 로드
    print("Loading DECA model...")
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    
    print("Loading AU detection model...")
    auconf = get_config()
    auconf.evaluate = True
    auconf.gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    set_env(auconf)
    AU_net = MEFARG(num_main_classes=auconf.num_main_classes, 
                    num_sub_classes=auconf.num_sub_classes,
                    backbone=auconf.arc).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()
    
    # 데이터셋 준비
    # 실제로는 datasets2.TestData를 사용하여 이미지를 로드하고
    # 별도로 레이블을 로드하는 방식이 더 나을 수 있습니다
    
    print("Loading DISFA dataset...")
    # DISFA 레이블 로드
    label_path = os.path.join(args.disfa_root, 'ActionUnit_Labels')
    image_paths = []
    labels = []
    
    for seq in TEST_DISFA_Sequence_split:
        seq_path = os.path.join(label_path, seq)
        au1_path = os.path.join(seq_path, f'{seq}_au1.txt')
        
        if not os.path.exists(au1_path):
            print(f"Warning: {au1_path} not found, skipping {seq}")
            continue
        
        with open(au1_path, 'r') as f:
            total_frames = len(f.readlines())
        
        au_label_array = np.zeros((total_frames, len(DISFA_AU_IDX)), dtype=np.int32)
        
        for ai, au in enumerate(DISFA_AU_IDX):
            au_label_path = os.path.join(seq_path, f'{seq}_au{au}.txt')
            if not os.path.exists(au_label_path):
                continue
            
            with open(au_label_path, 'r') as f:
                for t, line in enumerate(f.readlines()):
                    frame_idx, au_intensity = line.strip().split(',')
                    au_intensity = int(au_intensity)
                    au_label_array[t, ai] = 1 if au_intensity >= 1 else 0
        
        # 이미지 경로와 레이블 저장
        for frame_idx in range(total_frames):
            img_path = os.path.join(args.disfa_root, 'video', seq, 'croppedImages2', f'{frame_idx}.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(args.disfa_root, 'video', seq, 'originalImages', f'{frame_idx}.png')
            
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(au_label_array[frame_idx])
    
    print(f"Loaded {len(image_paths)} images")
    
    # 예측 수행
    print("Running predictions...")
    all_preds = []
    all_labels = []
    
    # 각 시퀀스별로 처리 (윈도우 기반 처리)
    K = 3  # 윈도우 크기
    half = K // 2
    
    for seq in TEST_DISFA_Sequence_split:
        inputpath = os.path.join(args.disfa_root, 'video', seq, 'croppedImages2')
        if not os.path.exists(inputpath):
            inputpath = os.path.join(args.disfa_root, 'video', seq, 'originalImages')
        
        if not os.path.exists(inputpath):
            continue
        
        # 이미지 데이터 로드
        testdata = datasets.TestData(inputpath, iscrop=args.iscrop, 
                                    crop_size=deca_cfg.dataset.image_size, scale=1.1)
        
        # 레이블 로드
        seq_path = os.path.join(label_path, seq)
        au1_path = os.path.join(seq_path, f'{seq}_au1.txt')
        if not os.path.exists(au1_path):
            continue
        
        with open(au1_path, 'r') as f:
            total_frames = len(f.readlines())
        
        seq_labels = np.zeros((total_frames, len(DISFA_AU_IDX)), dtype=np.int32)
        for ai, au in enumerate(DISFA_AU_IDX):
            au_label_path = os.path.join(seq_path, f'{seq}_au{au}.txt')
            if not os.path.exists(au_label_path):
                continue
            
            with open(au_label_path, 'r') as f:
                for t, line in enumerate(f.readlines()):
                    frame_idx, au_intensity = line.strip().split(',')
                    au_intensity = int(au_intensity)
                    seq_labels[t, ai] = 1 if au_intensity >= 1 else 0
        
        # 예측 수행
        for i in tqdm(range(half, len(testdata) - half), desc=f"Processing {seq}"):
            if i >= len(seq_labels):
                break
            
            # 윈도우 이미지 준비
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]
            
            images = torch.cat((data_1['image'][None, ...], 
                              data['image'][None, ...], 
                              data_3['image'][None, ...]), 0).to(device)
            
            with torch.no_grad():
                # DECA 인코딩
                codedict_old, codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)
                
                # AU 예측 (중앙 프레임 사용)
                image_au = AU_net(images[1:2])
                
                # 출력 형식에 따라 처리
                if isinstance(image_au, tuple):
                    au_pred = image_au[1]  # sub_output 사용
                else:
                    au_pred = image_au
                
                pred = (torch.sigmoid(au_pred) >= args.threshold).long().cpu().numpy()[0]
                label = seq_labels[i].astype(np.int32)
                
                all_preds.append(pred)
                all_labels.append(label)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print(f"Total predictions: {len(all_preds)}")
    
    # Confusion matrix 생성
    print("Creating confusion matrices...")
    confusion_matrices = []
    for au_idx in range(len(DISFA_AU_IDX)):
        cm = confusion_matrix_numpy(all_labels[:, au_idx], all_preds[:, au_idx], labels=[0, 1])
        confusion_matrices.append(cm)
    
    # 결과 출력
    print_metrics(confusion_matrices, DISFA_AU_NAMES)
    
    # 시각화
    plot_au_confusion_matrix(confusion_matrices, DISFA_AU_NAMES, 
                            save_path=args.save_path)
    
    print(f"\nConfusion matrix saved to {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DISFA AU Confusion Matrix')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    parser.add_argument('--disfa_root', 
                       default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA',
                       type=str, help='DISFA 데이터셋 루트 경로')
    parser.add_argument('--pretrained_modelpath_ViT',
                       default='/media/cine/First/HWPJ2/DetailNew/model.tar',
                       type=str, help='DECA 모델 경로')
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str)
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--threshold', default=0.5, type=float, help='AU 이진화 임계값')
    parser.add_argument('--save_path', default='./disfa_confusion_matrix.png', 
                       type=str, help='저장 경로')
    
    # AU 모델 설정은 conf.py에서 가져옴
    main(parser.parse_args())


