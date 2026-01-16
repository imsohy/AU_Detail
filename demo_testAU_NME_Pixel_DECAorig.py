"""
AFEW-VA 데이터셋 전용 Pixel-by-Pixel NME (Normalized Mean Error) 계산 스크립트
원본 이미지와 원본 DECA 모델이 생성한 이미지 간의 픽셀 단위 NME를 계산합니다.

이 스크립트는 gatfarec_Video_DetailNew_20260103.py에서 제공하는 원본 DECA 결과(_old 접미사)를 사용합니다.

AFEW-VA 데이터셋 구조:
- 입력 이미지: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg

변경사항:
- TestDataRecursive 클래스: recursive하게 하위 디렉토리에서 이미지 검색
- Pixel-by-Pixel NME 계산: 원본 이미지와 원본 DECA 생성 이미지 간의 픽셀 단위 오차 계산
- 원본 DECA 결과 사용: shape_detail_images_old, shape_images_old 등 _old 접미사가 붙은 결과 사용
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
from decalib.gatfarec_Video_DetailNew_20260103 import DECA
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


def calculate_pixel_nme(original_image, generated_image, 
                        normalization='image_size', 
                        pixel_range=1.0,
                        error_type='l2'):
    """
    Calculate Pixel-by-Pixel Normalized Mean Error (NME)
    
    Args:
        original_image: [C, H, W] or [H, W, C] tensor/numpy - 원본 이미지
        generated_image: [C, H, W] or [H, W, C] tensor/numpy - 생성 이미지
        normalization: 'image_size' (기본값), 'pixel_range', 'diagonal', 'mean_intensity', 'max_intensity'
        pixel_range: 픽셀 값 범위 (1.0 for 0-1, 255.0 for 0-255)
        error_type: 'l2' (Euclidean), 'l1' (Manhattan)
    
    Returns:
        nme: float - Pixel Normalized Mean Error
        pixel_errors: [H, W] numpy array - 각 픽셀의 오차 맵
    """
    # Convert to numpy if tensor
    if torch.is_tensor(original_image):
        orig = original_image.detach().cpu().numpy()
    else:
        orig = np.array(original_image)
    
    if torch.is_tensor(generated_image):
        gen = generated_image.detach().cpu().numpy()
    else:
        gen = np.array(generated_image)
    
    # Ensure same shape
    if orig.shape != gen.shape:
        raise ValueError(f"Shape mismatch: orig={orig.shape}, gen={gen.shape}")
    
    # Normalize to [H, W, C] format if needed
    if orig.ndim == 3:
        if orig.shape[0] == 3 or orig.shape[0] == 1:  # [C, H, W]
            orig = orig.transpose(1, 2, 0)
            gen = gen.transpose(1, 2, 0)
        elif orig.shape[2] == 3 or orig.shape[2] == 1:  # [H, W, C]
            pass  # Already in correct format
        else:
            raise ValueError(f"Unexpected image shape: {orig.shape}")
    
    H, W, C = orig.shape
    
    # Calculate pixel-wise error
    if error_type == 'l2':
        # L2 distance (Euclidean)
        pixel_errors = np.linalg.norm(orig - gen, axis=-1)  # [H, W]
    elif error_type == 'l1':
        # L1 distance (Manhattan)
        pixel_errors = np.abs(orig - gen).sum(axis=-1)  # [H, W]
    else:
        raise ValueError(f"Unknown error type: {error_type}")
    
    mean_error = pixel_errors.mean()
    
    # Normalization factor
    if normalization == 'image_size':
        # Normalize by image size (sqrt of total pixels)
        norm_factor = np.sqrt(H * W)
    
    elif normalization == 'pixel_range':
        # Normalize by pixel value range
        norm_factor = pixel_range
    
    elif normalization == 'diagonal':
        # Normalize by image diagonal
        norm_factor = np.sqrt(H**2 + W**2)
    
    elif normalization == 'mean_intensity':
        # Normalize by mean intensity of original image
        norm_factor = orig.mean() * pixel_range
        if norm_factor < 1e-6:
            norm_factor = pixel_range  # Fallback
    
    elif normalization == 'max_intensity':
        # Normalize by max intensity of original image
        norm_factor = orig.max() * pixel_range
        if norm_factor < 1e-6:
            norm_factor = pixel_range  # Fallback
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    # Avoid division by zero
    norm_factor = max(norm_factor, 1e-6)
    nme = mean_error / norm_factor
    
    return nme, pixel_errors


def extract_generated_image(opdict, visdict, image_type='shape_detail_images_old', frame_idx=1):
    """
    opdict 또는 visdict에서 생성된 이미지 추출 (원본 DECA 결과용)
    
    Args:
        opdict: DECA decode 결과의 opdict
        visdict: DECA decode 결과의 visdict
        image_type: 'shape_detail_images_old', 'shape_images_old', 
                   'render_images_old', 'rendered_images_old' 등 원본 DECA 결과
        frame_idx: 사용할 프레임 인덱스 (0, 1, 2 중 middle frame은 1)
    
    Returns:
        generated_image: [C, H, W] tensor
    """
    # visdict 또는 opdict에서 텐서 가져오기 (인덱싱 없이 먼저 가져오기)
    if image_type in visdict:
        image_tensor = visdict[image_type]  # [B, C, H, W] 또는 [1, C, H, W]
    elif image_type in opdict:
        image_tensor = opdict[image_type]  # [B, C, H, W] 또는 [1, C, H, W]
    else:
        available_keys = list(visdict.keys()) + list(opdict.keys())
        raise ValueError(f"Image type '{image_type}' not found in opdict or visdict. "
                        f"Available keys: {available_keys}")
    
    # 텐서의 첫 번째 차원 크기 확인
    batch_size = image_tensor.shape[0]
    
    # 배치 크기에 따라 적절한 인덱스 선택 (항상 middle frame 우선)
    if batch_size == 1:
        # 배치 크기가 1이면 인덱스 0만 사용 가능 (유일한 프레임 = middle)
        selected_idx = 0
    elif batch_size == 2:
        # 배치 크기가 2면 인덱스 1 사용 (middle frame)
        selected_idx = 1
    else:
        # 배치 크기가 3 이상이면 middle frame 사용
        # batch_size가 3이면 인덱스 1, 5면 인덱스 2 (중간)
        middle_idx = batch_size // 2
        # frame_idx가 범위 내에 있으면 사용, 아니면 middle_idx 사용
        if 0 <= frame_idx < batch_size:
            selected_idx = frame_idx
        else:
            selected_idx = middle_idx  # middle frame 사용
    
    generated_image = image_tensor[selected_idx]  # [C, H, W]
    
    return generated_image


def main(args):
    base_savefolder = args.savefolder
    device = args.device
    
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
    print(f"Base input path: {base_inputpath}")
    print(f"Image type (Original DECA): {args.image_type}")
    print(f"Pixel normalization: {args.pixel_normalization}")
    print(f"Error type: {args.error_type}")
    
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
        
        os.makedirs(os.path.join(savefolder, 'error_maps'), exist_ok=True)
        
        # NME 계산을 위한 변수
        nme_list = []
        valid_count = 0
        skipped_count = 0
        
        for i in tqdm(range(1, len(testdata)-1), desc=f"Processing {dir_name}"):
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]
            name = data['imagename']
            imagepath = data['imagepath']  # 전체 경로 가져오기
            
            # 서브디렉토리 정보 추출
            # 예: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg
            #     inputpath = /home/cine/Downloads/AFEW-VA/croppedImages/01
            #     -> rel_path = 001/00001.jpg
            rel_path = os.path.relpath(imagepath, inputpath)  # inputpath 기준 상대 경로
            rel_dir = os.path.dirname(rel_path)  # 001 (서브디렉토리명)
            filename_without_ext = os.path.splitext(os.path.basename(rel_path))[0]  # 00001
            
            # 서브디렉토리 정보를 포함한 이름 생성
            if rel_dir and rel_dir != '.':
                # 서브디렉토리가 있는 경우: 001_00001
                error_map_name = f"{rel_dir}_{filename_without_ext}"
            else:
                # 서브디렉토리가 없는 경우: 00001 (기존과 동일)
                error_map_name = filename_without_ext
            
            images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]), 0).to(device)
            
            with torch.no_grad():
                codedict_old, codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True)
            
            # 원본 이미지 추출 (middle frame)
            original_image = data['image']  # [C, H, W] tensor, 0-1 범위
            
            try:
                # 원본 DECA 생성 이미지 추출
                generated_image = extract_generated_image(
                    opdict, visdict, 
                    image_type=args.image_type,
                    frame_idx=1  # middle frame
                )
                
                # Pixel NME 계산
                nme, pixel_errors = calculate_pixel_nme(
                    original_image,
                    generated_image,
                    normalization=args.pixel_normalization,
                    pixel_range=1.0,  # 0-1 범위
                    error_type=args.error_type
                )
                
                nme_list.append(nme)
                valid_count += 1
                
                # Error map 저장 (옵션)
                if args.saveErrorMap:
                    error_map_path = os.path.join(savefolder, 'error_maps', error_map_name + '_error_map.npy')
                    np.save(error_map_path, pixel_errors)
                    
                    # 시각화용 이미지로도 저장
                    if pixel_errors.max() > 0:
                        error_map_vis = (pixel_errors / pixel_errors.max() * 255).astype(np.uint8)
                    else:
                        error_map_vis = np.zeros_like(pixel_errors, dtype=np.uint8)
                    cv2.imwrite(os.path.join(savefolder, 'error_maps', error_map_name + '_error_map.jpg'), 
                               error_map_vis)
                
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
            
            print(f"\nPixel NME Results for {dir_name} (Original DECA):")
            print(f"  Mean NME: {mean_nme:.6f}")
            print(f"  Std NME:  {std_nme:.6f}")
            print(f"  Min NME:  {min_nme:.6f}")
            print(f"  Max NME:  {max_nme:.6f}")
            print(f"  Valid samples: {valid_count}")
            print(f"  Skipped samples: {skipped_count}")
            
            # 결과를 파일로 저장
            result_file = os.path.join(savefolder, 'pixel_nme_results_DECAorig.json')
            with open(result_file, 'w') as f:
                json.dump({
                    'mean_nme': float(mean_nme),
                    'std_nme': float(std_nme),
                    'min_nme': float(min_nme),
                    'max_nme': float(max_nme),
                    'valid_count': int(valid_count),
                    'skipped_count': int(skipped_count),
                    'pixel_normalization': args.pixel_normalization,
                    'error_type': args.error_type,
                    'image_type': args.image_type,
                    'model_type': 'Original DECA'
                }, f, indent=2)
            
            # 텍스트 파일로도 저장
            result_txt_file = os.path.join(savefolder, 'pixel_nme_results_DECAorig.txt')
            with open(result_txt_file, 'w') as f:
                f.write("Pixel-by-Pixel NME (Normalized Mean Error) Calculation Results - Original DECA - AFEW-VA Dataset\n")
                f.write("=" * 60 + "\n")
                f.write(f"Directory: {dir_name}\n")
                f.write(f"Model Type: Original DECA\n")
                f.write(f"Image type: {args.image_type}\n")
                f.write(f"Pixel normalization method: {args.pixel_normalization}\n")
                f.write(f"Error type: {args.error_type}\n")
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
        print("Overall Pixel NME Results Summary - Original DECA - AFEW-VA Dataset")
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
        summary_file = os.path.join(base_savefolder, 'pixel_nme_summary_DECAorig.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'dataset': 'AFEW-VA',
                'model_type': 'Original DECA',
                'overall_mean_nme': float(overall_mean),
                'overall_std_nme': float(overall_std),
                'total_valid_samples': int(total_valid),
                'total_skipped_samples': int(total_skipped),
                'pixel_normalization': args.pixel_normalization,
                'error_type': args.error_type,
                'image_type': args.image_type,
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
    parser = argparse.ArgumentParser(description='DECA: Pixel-by-Pixel NME Calculation for Original DECA Results - AFEW-VA Dataset')
    parser.add_argument('-i', '--inputpath', 
                        default='/home/cine/Downloads/AFEW-VA/croppedImages/', 
                        type=str,
                        help='path to AFEW-VA croppedImages directory (e.g., /home/cine/Downloads/AFEW-VA/croppedImages/)')
    parser.add_argument('-s', '--savefolder', 
                        default='/media/cine/First/HWPJ2/ProjectResult/Demos/PixelNME_AFEW_Evaluation_DECAorig/', 
                        type=str, 
                        help='path to the output directory')
    parser.add_argument('--pretrained_modelpath_ViT',
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune_2/model.tar', 
                        type=str,
                        help='model.tar path')
    
    # Pixel NME 관련 인자들 - 원본 DECA 결과 옵션
    parser.add_argument('--image_type', 
                        default='shape_detail_images_old', 
                        type=str,
                        choices=['shape_detail_images_old', 'shape_images_old', 
                                'render_images_old', 'rendered_images_old',
                                'shape_detail_images_full_old', 'shape_images_full_old'],
                        help='Type of generated image to compare (Original DECA results with _old suffix)')
    parser.add_argument('--pixel_normalization', 
                        default='image_size', 
                        type=str,
                        choices=['image_size', 'pixel_range', 'diagonal', 'mean_intensity', 'max_intensity'],
                        help='Normalization method for pixel NME calculation')
    parser.add_argument('--error_type', 
                        default='l2', 
                        type=str,
                        choices=['l1', 'l2'],
                        help='Error type: l1 (Manhattan) or l2 (Euclidean)')
    parser.add_argument('--saveErrorMap', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save pixel error maps')
    
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
    parser.add_argument('--verbose', 
                        default=False, 
                        type=lambda x: x.lower() in ['true', '1'],
                        help='whether to print verbose messages')
    main(parser.parse_args())
