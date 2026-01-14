"""
AFEW-VA 데이터셋 전용 NME (Normalized Mean Error) 계산 스크립트 (v4)
학습된 모델을 사용하여 landmark를 추출하고 GT landmark와 비교하여 NME를 계산합니다.

AFEW-VA 데이터셋 구조:
- 입력 이미지: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg
- GT landmark: /home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy

v4 변경사항:
- TestDataRecursive 클래스 추가: recursive하게 하위 디렉토리에서 이미지 검색
- AFEW-VA 구조 (/croppedImages/01/001/...)에 맞게 모든 하위 디렉토리에서 이미지 수집
"""
# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import cv2
import numpy as np
from time import time
import argparse
from tqdm import tqdm
import torch
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from decalib.gatfarec_Video_DetailNew_20260104 import DECA
from decalib.datasets import datasets2 as datasets
from decalib.utils import util
from decalib.utils.config_wt_DetailNew_20260103 import cfg as deca_cfg
from skimage.io import imread
from skimage.transform import estimate_transform, warp


class TestDataRecursive(datasets.TestData):
    """
    AFEW-VA 구조에 맞게 recursive하게 이미지를 찾는 TestData 클래스
    기존 TestData를 상속받아 __init__만 수정하여 하위 디렉토리에서도 이미지를 찾도록 함
    """
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.05, face_detector='retinaface',
                 ifCenter='', ifSize=''):
        '''
        testpath: folder, imagepath_list, image path, video path
        AFEW-VA 구조: /croppedImages/01/001/00001.jpg 처럼 하위 디렉토리에 이미지가 있음
        '''
        # mediapipe_idx는 부모 클래스와 동일
        self.mediapipe_idx = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,  55,  63,  65,  66,  70,
        105, 107, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
         7,  33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 168,   6,
        197, 195,   5, 4, 129,  98,  97,   2, 326, 327, 358,   0,  13,  14,  17,  37,  39,  40,
         61,  78,  80, 81,  82,  84,  87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270,
        291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        # Recursive glob 패턴 사용하여 모든 하위 디렉토리에서 이미지 찾기
        if os.path.isdir(testpath):
            # Recursive glob: **/*.jpg 패턴으로 모든 하위 디렉토리 검색
            self.imagepath_list = (
                glob.glob(os.path.join(testpath, '**/*.jpg'), recursive=True) +
                glob.glob(os.path.join(testpath, '**/*.png'), recursive=True) +
                glob.glob(os.path.join(testpath, '**/*.bmp'), recursive=True)
            )
        else:
            # testpath가 디렉토리가 아닌 경우 기존 로직 사용 (fallback)
            self.imagepath_list = glob.glob(testpath + '/*.jpg') + glob.glob(testpath + '/*.png') + glob.glob(testpath + '/*.bmp')
        
        self.type = 'image'

        # 이미지 경로 정렬 (기존 로직과 동일)
        try:
            self.imagepath_list = sorted(self.imagepath_list,
                                         key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0].split('frame')[-1]))
        except:
            self.imagepath_list = sorted(self.imagepath_list)

        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size


def calculate_nme(predicted_landmarks, gt_landmarks, normalization='inter-ocular', landmark_format='68'):
    """
    Calculate Normalized Mean Error (NME)
    
    Args:
        predicted_landmarks: [N, K, 2] or [K, 2] tensor/numpy - 예측된 landmark
        gt_landmarks: [N, K, 2] or [K, 2] tensor/numpy - GT landmark
        normalization: 'inter-ocular' (기본값), 'inter-pupil', 'bbox', 'face_size'
        landmark_format: '68' (FLAME 68-point) or 'mediapipe'
    
    Returns:
        nme: float - Normalized Mean Error
    """
    # Convert to numpy if tensor
    if torch.is_tensor(predicted_landmarks):
        pred = predicted_landmarks.detach().cpu().numpy()
    else:
        pred = np.array(predicted_landmarks)
    
    if torch.is_tensor(gt_landmarks):
        gt = gt_landmarks.detach().cpu().numpy()
    else:
        gt = np.array(gt_landmarks)
    
    # Ensure 2D shape [N, K, 2]
    if pred.ndim == 2:
        pred = pred[None, ...]
    if gt.ndim == 2:
        gt = gt[None, ...]
    
    # Ensure same batch size
    if pred.shape[0] != gt.shape[0]:
        raise ValueError(f"Batch size mismatch: pred={pred.shape[0]}, gt={gt.shape[0]}")
    
    batch_size = pred.shape[0]
    nme_list = []
    
    for i in range(batch_size):
        pred_i = pred[i]  # [K, 2]
        gt_i = gt[i]      # [K, 2]
        
        # Calculate Euclidean distance per landmark
        errors = np.linalg.norm(pred_i - gt_i, axis=-1)  # [K]
        mean_error = errors.mean()
        
        # Normalization factor
        if normalization == 'inter-ocular':
            # For 68-point FLAME: left eye 36, right eye 45
            if landmark_format == '68':
                left_eye_idx, right_eye_idx = 36, 45
            else:  # mediapipe or other formats
                # Approximate eye center indices (may need adjustment)
                left_eye_idx, right_eye_idx = 36, 45  # Default to 68-point indices
            
            if gt_i.shape[0] > max(left_eye_idx, right_eye_idx):
                left_eye = gt_i[left_eye_idx, :]
                right_eye = gt_i[right_eye_idx, :]
                norm_factor = np.linalg.norm(left_eye - right_eye)
            else:
                # 경고를 출력하고 샘플을 스킵
                if args.verbose:
                    print(f"Warning: Eye indices invalid, skipping sample (landmark count: {gt_i.shape[0]})")
                return None  # 또는 continue로 스킵
        
        elif normalization == 'inter-pupil':
            # Similar to inter-ocular but using pupil centers
            # For 68-point, use same indices as inter-ocular
            if landmark_format == '68':
                left_eye_idx, right_eye_idx = 36, 45
            else:
                left_eye_idx, right_eye_idx = 36, 45
            
            if gt_i.shape[0] > max(left_eye_idx, right_eye_idx):
                left_eye = gt_i[left_eye_idx, :]
                right_eye = gt_i[right_eye_idx, :]
                norm_factor = np.linalg.norm(left_eye - right_eye)
            else:
                bbox_min = gt_i.min(axis=0)
                bbox_max = gt_i.max(axis=0)
                norm_factor = np.linalg.norm(bbox_max - bbox_min)
        
        elif normalization == 'bbox':
            # Use bounding box diagonal
            bbox_min = gt_i.min(axis=0)
            bbox_max = gt_i.max(axis=0)
            norm_factor = np.linalg.norm(bbox_max - bbox_min)
        
        elif normalization == 'face_size':
            # Use face width or height
            bbox_min = gt_i.min(axis=0)
            bbox_max = gt_i.max(axis=0)
            face_size = np.max(bbox_max - bbox_min)
            norm_factor = face_size
        
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
        
        # Avoid division by zero
        norm_factor = max(norm_factor, 1e-6)
        nme = mean_error / norm_factor
        nme_list.append(nme)
    
    return np.mean(nme_list) if len(nme_list) > 0 else 0.0


def get_gt_landmark_path_afew(imagepath, base_inputpath, gt_landmark_dir):
    """
    AFEW-VA 데이터셋 구조에 맞게 GT landmark 경로를 생성합니다.
    
    AFEW-VA 구조:
    - 이미지: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg
    - GT:     /home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy
    
    Args:
        imagepath: 이미지의 전체 경로 (data['imagepath']에서 얻음)
        base_inputpath: 입력 이미지 기본 경로 (예: /home/cine/Downloads/AFEW-VA/croppedImages)
        gt_landmark_dir: GT landmark 디렉토리 패턴 (예: "/home/cine/Downloads/AFEW-VA/lmkGT_N/*")
    
    Returns:
        gt_landmark_path: GT landmark 파일 경로
    """
    if imagepath is None:
        return None
    
    # 경로 정규화 (Windows/Linux 호환)
    imagepath = imagepath.replace('\\', '/')
    base_inputpath = base_inputpath.replace('\\', '/')
    
    # base_inputpath 기준으로 상대 경로 추출
    # 예: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg
    #     -> 01/001/00001
    try:
        # base_inputpath를 기준으로 상대 경로 계산
        relative_path = os.path.relpath(imagepath, base_inputpath)
        relative_path = os.path.splitext(relative_path)[0]  # 확장자 제거
        # 경로 구분자를 정규화
        relative_path = relative_path.replace('\\', '/')
    except ValueError:
        # 상대 경로를 만들 수 없는 경우 (다른 드라이브 등)
        # 경로에서 AFEW-VA 구조 부분 추출
        # 예: .../croppedImages/01/001/00001.jpg -> 01/001/00001
        parts = imagepath.split('/')
        # 'croppedImages' 또는 'Images' 다음 부분 찾기
        try:
            idx = -1
            for i, part in enumerate(parts):
                if part in ['croppedImages', 'Images']:
                    idx = i
                    break
            if idx >= 0 and idx + 3 < len(parts):
                # croppedImages/Images 다음 3개 디렉토리 사용
                relative_path = '/'.join(parts[idx+1:idx+4])
                relative_path = os.path.splitext(relative_path)[0]
            else:
                # fallback: 마지막 3개 디렉토리 사용
                if len(parts) >= 3:
                    relative_path = '/'.join(parts[-3:])
                    relative_path = os.path.splitext(relative_path)[0]
                else:
                    # 최후의 수단: 파일명만 사용
                    relative_path = os.path.splitext(os.path.basename(imagepath))[0]
        except:
            # 최후의 수단: 파일명만 사용
            relative_path = os.path.splitext(os.path.basename(imagepath))[0]
    
    # GT landmark 경로 생성
    # 패턴: gt_landmark_dir/* -> gt_landmark_dir/relative_path.npy
    gt_landmark_path = gt_landmark_dir.replace("*", relative_path + ".npy")
    
    return gt_landmark_path


def main(args):
    base_savefolder = args.savefolder
    device = args.device
    
    # GT landmark 디렉토리 설정
    landmarkDir_GT = args.gt_landmark_dir
    
    # Get all subdirectories under the input path
    base_inputpath = args.inputpath.rstrip('/')
    
    if not os.path.isdir(base_inputpath):
        print(f"Error: Input path {base_inputpath} does not exist!")
        return
    
    # Get all subdirectories under the input path
    subdirs = [d for d in os.listdir(base_inputpath) 
               if os.path.isdir(os.path.join(base_inputpath, d))]
    subdirs = sorted(subdirs)
    
    if len(subdirs) == 0:
        print(f"Warning: No subdirectories found in {base_inputpath}")
        return
    
    print(f"Found {len(subdirs)} directories to process: {subdirs}")
    print(f"GT landmark directory pattern: {landmarkDir_GT}")
    print(f"Base input path: {base_inputpath}")
    
    # NME 결과 저장을 위한 변수
    all_nme_results = {}
    
    # Process each directory
    for dir_name in subdirs:
        inputpath = os.path.join(base_inputpath, dir_name)
        savefolder = os.path.join(base_savefolder, dir_name)
        os.makedirs(savefolder, exist_ok=True)
        
        print(f"\nProcessing directory: {dir_name}")
        print(f"Input path: {inputpath}")
        print(f"Save folder: {savefolder}")
        
        # Load test images using TestDataRecursive (recursive search)
        testdata = TestDataRecursive(inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                     scale=1.1)
        
        print(f"Found {len(testdata)} images in {dir_name} (including subdirectories)")
        
        if len(testdata) < 3:
            print(f"Warning: Directory {dir_name} has less than 3 images, skipping...")
            continue
        
        # Run DECA
        deca_cfg.model.use_tex = args.useTex
        deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
        deca_cfg.rasterizer_type = args.rasterizer_type
        deca_cfg.model.extract_tex = args.extractTex
        deca = DECA(config=deca_cfg, device=device)
        
        os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
        os.makedirs(os.path.join(savefolder, 'landmarks'), exist_ok=True)
        
        # NME 계산을 위한 변수
        nme_list = []
        valid_count = 0
        skipped_count = 0
        
        for i in tqdm(range(1, len(testdata)-1), desc=f"Processing {dir_name}"):
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]
            name = data['imagename']
            
            # imagepath 가져오기 (AFEW-VA 구조 추출용)
            imagepath = data.get('imagepath', None)
            
            images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]), 0).to(device)
            
            with torch.no_grad():
                codedict_old, codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True)
            
            # 예측된 landmark 추출 (middle frame만 사용)
            predicted_landmarks = opdict['landmarks2d'][1]  # [K, 2] - middle frame의 landmark
            
            # GT landmark 로드 (AFEW-VA 구조 사용)
            gt_landmark_path = get_gt_landmark_path_afew(imagepath, base_inputpath, landmarkDir_GT)
            
            if gt_landmark_path is None or not os.path.exists(gt_landmark_path):
                skipped_count += 1
                if args.verbose:
                    print(f"Warning: GT landmark not found for {name}")
                    if gt_landmark_path:
                        print(f"  Expected path: {gt_landmark_path}")
                continue
            
            try:
                landmarks2dGT = np.load(gt_landmark_path, allow_pickle=True)
                
                # GT가 3D인 경우 2D만 사용
                if landmarks2dGT.ndim == 2:
                    if landmarks2dGT.shape[1] >= 2:
                        landmarks2dGT_2d = landmarks2dGT[:, :2] if landmarks2dGT.shape[1] > 2 else landmarks2dGT
                    else:
                        if args.verbose:
                            print(f"Warning: Invalid GT landmark shape for {name}, skipping...")
                        skipped_count += 1
                        continue
                else:
                    if args.verbose:
                        print(f"Warning: Unexpected GT landmark shape for {name}, skipping...")
                    skipped_count += 1
                    continue
                
                # Landmark 개수 확인 및 조정
                if predicted_landmarks.shape[0] != landmarks2dGT_2d.shape[0]:
                    if args.verbose:
                        print(f"Warning: Landmark count mismatch for {name}: "
                              f"pred={predicted_landmarks.shape[0]}, gt={landmarks2dGT_2d.shape[0]}, skipping...")
                    skipped_count += 1
                    continue
                
                # NME 계산
                nme = calculate_nme(
                    predicted_landmarks.cpu().numpy(),
                    landmarks2dGT_2d,
                    normalization=args.normalization,
                    landmark_format='68'
                )
                
                nme_list.append(nme)
                valid_count += 1
                
                # Landmark 저장 (옵션)
                if args.saveKpt:
                    np.save(os.path.join(savefolder, 'landmarks', name + '_pred.npy'), 
                           predicted_landmarks.cpu().numpy())
                    np.save(os.path.join(savefolder, 'landmarks', name + '_gt.npy'), 
                           landmarks2dGT_2d)
                
            except Exception as e:
                if args.verbose:
                    print(f"Error processing {name}: {e}")
                skipped_count += 1
                continue
        
        # 결과 저장
        if valid_count > 0:
            mean_nme = np.mean(nme_list)
            std_nme = np.std(nme_list)
            min_nme = np.min(nme_list)
            max_nme = np.max(nme_list)
            
            all_nme_results[dir_name] = {
                'mean_nme': mean_nme,
                'std_nme': std_nme,
                'min_nme': min_nme,
                'max_nme': max_nme,
                'valid_count': valid_count,
                'skipped_count': skipped_count,
                'all_nme': nme_list
            }
            
            print(f"\nNME Results for {dir_name}:")
            print(f"  Mean NME: {mean_nme:.6f}")
            print(f"  Std NME:  {std_nme:.6f}")
            print(f"  Min NME:  {min_nme:.6f}")
            print(f"  Max NME:  {max_nme:.6f}")
            print(f"  Valid samples: {valid_count}")
            print(f"  Skipped samples: {skipped_count}")
            
            # 결과를 파일로 저장
            result_file = os.path.join(savefolder, 'nme_results.json')
            with open(result_file, 'w') as f:
                json.dump({
                    'mean_nme': float(mean_nme),
                    'std_nme': float(std_nme),
                    'min_nme': float(min_nme),
                    'max_nme': float(max_nme),
                    'valid_count': int(valid_count),
                    'skipped_count': int(skipped_count),
                    'normalization': args.normalization
                }, f, indent=2)
            
            # 텍스트 파일로도 저장
            result_txt_file = os.path.join(savefolder, 'nme_results.txt')
            with open(result_txt_file, 'w') as f:
                f.write("NME (Normalized Mean Error) Calculation Results - AFEW-VA Dataset\n")
                f.write("=" * 60 + "\n")
                f.write(f"Directory: {dir_name}\n")
                f.write(f"Normalization method: {args.normalization}\n")
                f.write(f"Valid samples: {valid_count}\n")
                f.write(f"Skipped samples: {skipped_count}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Mean NME: {mean_nme:.6f}\n")
                f.write(f"Std NME:  {std_nme:.6f}\n")
                f.write(f"Min NME:  {min_nme:.6f}\n")
                f.write(f"Max NME:  {max_nme:.6f}\n")
                f.write("=" * 60 + "\n")
        else:
            print(f"\nWarning: No valid samples found for {dir_name}")
    
    # 전체 결과 요약
    if all_nme_results:
        print(f"\n{'='*60}")
        print("Overall NME Results Summary - AFEW-VA Dataset")
        print(f"{'='*60}")
        
        overall_mean = np.mean([r['mean_nme'] for r in all_nme_results.values()])
        overall_std = np.std([r['mean_nme'] for r in all_nme_results.values()])
        total_valid = sum([r['valid_count'] for r in all_nme_results.values()])
        total_skipped = sum([r['skipped_count'] for r in all_nme_results.values()])
        
        print(f"Overall Mean NME: {overall_mean:.6f}")
        print(f"Overall Std NME:  {overall_std:.6f}")
        print(f"Total valid samples: {total_valid}")
        print(f"Total skipped samples: {total_skipped}")
        
        # 전체 결과 저장
        summary_file = os.path.join(base_savefolder, 'nme_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'dataset': 'AFEW-VA',
                'overall_mean_nme': float(overall_mean),
                'overall_std_nme': float(overall_std),
                'total_valid_samples': int(total_valid),
                'total_skipped_samples': int(total_skipped),
                'normalization': args.normalization,
                'per_directory': {k: {
                    'mean_nme': float(v['mean_nme']),
                    'std_nme': float(v['std_nme']),
                    'valid_count': int(v['valid_count']),
                    'skipped_count': int(v['skipped_count'])
                } for k, v in all_nme_results.items()}
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")
    
    print(f'\nAll directories processed. Results saved in {base_savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: NME Calculation for AFEW-VA Dataset (v4 with recursive search)')
    parser.add_argument('-i', '--inputpath', 
                        default='/home/cine/Downloads/AFEW-VA/croppedImages/', 
                        type=str,
                        help='path to AFEW-VA croppedImages directory (e.g., /home/cine/Downloads/AFEW-VA/croppedImages/)')
    parser.add_argument('-s', '--savefolder', 
                        default='/media/cine/First/HWPJ2/ProjectResult/Demos/NME_AFEW_Evaluation/', 
                        type=str, 
                        help='path to the output directory')
    parser.add_argument('--pretrained_modelpath_ViT',
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune_2/model.tar', 
                        type=str,
                        help='model.tar path')
    parser.add_argument('--gt_landmark_dir', 
                        default='/home/cine/Downloads/AFEW-VA/lmkGT_N/*', 
                        type=str,
                        help='GT landmark directory pattern (use * as wildcard, e.g., /home/cine/Downloads/AFEW-VA/lmkGT_N/*)')
    parser.add_argument('--normalization', 
                        default='inter-ocular', 
                        type=str,
                        choices=['inter-ocular', 'inter-pupil', 'bbox', 'face_size'],
                        help='Normalization method for NME calculation')
    default_device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') is not None else 'cuda:1'
    parser.add_argument('--device', 
                        default=default_device, 
                        type=str,
                        help='set device, cpu for using cpu')
    parser.add_argument('--iscrop', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image (usually False for croppedImages)')
    parser.add_argument('--detector', 
                        default='retinaface', 
                        type=str,
                        help='detector for cropping face')
    parser.add_argument('--rasterizer_type', 
                        default='pytorch3d', 
                        type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--useTex', 
                        default=True, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model')
    parser.add_argument('--extractTex', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image')
    parser.add_argument('--saveKpt', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save predicted and GT landmarks')
    parser.add_argument('--verbose', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to print verbose messages')
    main(parser.parse_args())
