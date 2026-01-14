"""
AFEW-VA 데이터셋 전용 NME (Normalized Mean Error) 계산 스크립트
학습된 모델을 활용하여 예측된 랜드마크와 GT 랜드마크를 비교하여 NME를 계산합니다.

예시:
입력 이미지: /home/cine/Downloads/AFEW-VA/croppedImages/01/001/00001.jpg
GT 랜드마크: /home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy

주의:
AFEW-VA croppedImages는 보통 폴더 구조가 한 단계 더 깊습니다.
예: croppedImages/01/001/00001.jpg
datasets.TestData가 재귀 탐색을 하지 않는 경우, croppedImages/01에 이미지가 없다고 판단되어
전부 스킵될 수 있어, "이미지가 실제로 들어있는 leaf 폴더"들을 찾아서 처리하도록 수정했습니다.
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from decalib.gatfarec_Video_DetailNew_20260104 import DECA
from decalib.utils.config_wt_DetailNew_20260103 import cfg as deca_cfg
from decalib.utils import util

# datasets2 를 사용하고 있다고 가정 (원본 코드 유지)
import decalib.datasets.datasets2 as datasets


def calculate_nme(predicted_landmarks, gt_landmarks,
                  normalization='inter-ocular', landmark_format='68'):
    """
    NME (Normalized Mean Error) 계산

    Args:
        predicted_landmarks: 예측된 랜드마크 [K,2] or [N,K,2]
        gt_landmarks: GT 랜드마크 [K,2] or [N,K,2]
        normalization: 정규화 방식
            - 'inter-ocular': 양 눈 사이 거리(기본)
            - 'inter-pupil': 동공 사이 거리(동일 인덱스 사용)
            - 'bbox': bounding box 대각선
            - 'face_size': bounding box의 w/h 중 큰 값
        landmark_format: 랜드마크 포맷 ('68' 가정)

    Returns:
        nme: float
    """
    # numpy / torch 모두 처리
    if torch.is_tensor(predicted_landmarks):
        pred = predicted_landmarks.detach().cpu().numpy()
    else:
        pred = np.array(predicted_landmarks)

    if torch.is_tensor(gt_landmarks):
        gt = gt_landmarks.detach().cpu().numpy()
    else:
        gt = np.array(gt_landmarks)

    # shape 통일: [K,2]
    if pred.ndim == 3:
        pred = pred[0]
    if gt.ndim == 3:
        gt = gt[0]

    # Kx2 확인
    assert pred.shape == gt.shape, f"pred shape {pred.shape} != gt shape {gt.shape}"

    # point-wise error
    errors = np.linalg.norm(pred - gt, axis=1)  # [K]
    mean_error = np.mean(errors)

    # normalization factor
    if normalization in ['inter-ocular', 'inter-pupil']:
        # 68pt 기준: 36,45 를 눈의 대표점으로 사용 (정의가 다르면 수정 필요)
        if gt.shape[0] > 45:
            left_eye = gt[36]
            right_eye = gt[45]
            norm_factor = np.linalg.norm(left_eye - right_eye)
        else:
            # fallback: bbox diag
            min_xy = np.min(gt, axis=0)
            max_xy = np.max(gt, axis=0)
            norm_factor = np.linalg.norm(max_xy - min_xy)
    elif normalization == 'bbox':
        min_xy = np.min(gt, axis=0)
        max_xy = np.max(gt, axis=0)
        norm_factor = np.linalg.norm(max_xy - min_xy)
    elif normalization == 'face_size':
        min_xy = np.min(gt, axis=0)
        max_xy = np.max(gt, axis=0)
        w, h = (max_xy - min_xy)
        norm_factor = max(w, h)
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")

    # avoid zero
    norm_factor = max(norm_factor, 1e-8)
    nme = mean_error / norm_factor
    return float(nme)


def get_gt_landmark_path_afew(imagepath, base_inputpath, gt_landmark_dir):
    """
    이미지 경로로부터 GT 랜드마크 경로를 생성합니다.

    예:
      imagepath: /.../croppedImages/01/001/00001.jpg
      base_inputpath: /.../croppedImages
      gt_landmark_dir: /.../lmkGT_N/*

    => /.../lmkGT_N/01/001/00001.npy
    """
    if imagepath is None:
        return None

    # 상대 경로 추출
    relative_path = None
    try:
        relative_path = os.path.relpath(imagepath, base_inputpath)
        relative_path = os.path.splitext(relative_path)[0]  # remove extension
        relative_path = relative_path.replace('\\', '/')
    except Exception:
        # fallback: base_inputpath 문자열이 포함된 경우 split
        try:
            if base_inputpath in imagepath:
                relative_path = imagepath.split(base_inputpath)[-1].lstrip('/\\')
                relative_path = os.path.splitext(relative_path)[0]
                relative_path = relative_path.replace('\\', '/')
            else:
                # 최후의 수단: 파일명만
                relative_path = os.path.splitext(os.path.basename(imagepath))[0]
        except Exception:
            relative_path = os.path.splitext(os.path.basename(imagepath))[0]

    if relative_path is None:
        return None

    # 패턴 치환
    gt_landmark_path = gt_landmark_dir.replace("*", relative_path + ".npy")
    return gt_landmark_path


def main(args):
    base_savefolder = args.savefolder
    device = args.device

    # GT landmark 패턴
    landmarkDir_GT = args.gt_landmark_dir

    # 입력 루트
    base_inputpath = args.inputpath.rstrip('/')

    if not os.path.isdir(base_inputpath):
        print(f"Error: Input path {base_inputpath} does not exist!")
        return

    # ------------------------------------------------------------------
    # [PATCH] AFEW-VA croppedImages는 보통 구조가 한 단계 더 깊습니다:
    #   croppedImages/01/001/00001.jpg
    # 따라서 '이미지가 실제로 들어있는 leaf 폴더'들을 찾아서 처리합니다.
    # (datasets.TestData가 재귀 탐색을 하지 않는 경우를 대비)
    # ------------------------------------------------------------------
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    leaf_dirs = []

    for root, dirs, files in os.walk(base_inputpath):
        img_count = sum(f.lower().endswith(img_exts) for f in files)
        if img_count >= 3:
            leaf_dirs.append(root)

    leaf_dirs = sorted(leaf_dirs)

    if len(leaf_dirs) == 0:
        print(f"Warning: No leaf directories with >=3 images found under {base_inputpath}")
        print("  Check your dataset path or image extensions.")
        return

    print(f"Found {len(leaf_dirs)} leaf directories to process (each has >=3 images).")
    print(f"GT landmark directory pattern: {landmarkDir_GT}")
    print(f"Base input path: {base_inputpath}")

    # 전체 결과
    all_nme_results = {}

    # leaf 폴더 단위 처리
    for inputpath in leaf_dirs:
        # base_inputpath 기준 상대 경로 (예: "01/001")
        rel_dir = os.path.relpath(inputpath, base_inputpath).replace('\\', '/')
        savefolder = os.path.join(base_savefolder, rel_dir)
        os.makedirs(savefolder, exist_ok=True)

        print(f"\nProcessing directory: {rel_dir}")
        print(f"Input path: {inputpath}")
        print(f"Save folder: {savefolder}")

        # Load test images
        testdata = datasets.TestData(
            inputpath,
            iscrop=args.iscrop,
            crop_size=deca_cfg.dataset.image_size,
            scale=1.1
        )

        if len(testdata) < 3:
            print(f"Warning: Directory {rel_dir} has less than 3 images, skipping...")
            continue

        # Run DECA
        deca_cfg.model.use_tex = args.useTex
        deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
        deca_cfg.rasterizer_type = args.rasterizer_type
        deca_cfg.model.extract_tex = args.extractTex
        deca = DECA(config=deca_cfg, device=device)

        os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
        os.makedirs(os.path.join(savefolder, 'landmarks'), exist_ok=True)

        # NME 계산 변수
        nme_list = []
        skipped_count = 0
        valid_count = 0

        for i in tqdm(range(1, len(testdata) - 1), desc=f"Processing {rel_dir}"):
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]

            name = data.get('imagename', None)
            imagepath = data.get('imagepath', None)

            # 3 프레임 입력 구성
            images = torch.cat([data_1['image'], data['image'], data_3['image']], dim=0).to(device)

            with torch.no_grad():
                codedict_old, codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True)

            # predicted landmarks
            if 'landmarks2d' not in opdict:
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] No landmarks2d in opdict: {imagepath}")
                continue

            predicted_landmarks = opdict['landmarks2d'][0]  # [K,2] (torch)

            # GT landmark path
            gt_landmark_path = get_gt_landmark_path_afew(imagepath, base_inputpath, landmarkDir_GT)
            if gt_landmark_path is None or (not os.path.exists(gt_landmark_path)):
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] GT landmark not found: {gt_landmark_path} (image: {imagepath})")
                continue

            # Load GT landmarks
            try:
                landmarks2dGT = np.load(gt_landmark_path, allow_pickle=True)
            except Exception as e:
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] Failed to load GT landmark: {gt_landmark_path}, err={e}")
                continue

            # GT가 [K,3]이면 [K,2]만 사용
            if landmarks2dGT.ndim == 2 and landmarks2dGT.shape[1] >= 2:
                landmarks2dGT_2d = landmarks2dGT[:, :2]
            else:
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] Unexpected GT shape: {landmarks2dGT.shape} ({gt_landmark_path})")
                continue

            # pred/gt point count match check
            pred_np = predicted_landmarks.detach().cpu().numpy()
            if pred_np.shape[0] != landmarks2dGT_2d.shape[0]:
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] Landmark count mismatch: pred {pred_np.shape[0]} vs gt {landmarks2dGT_2d.shape[0]}")
                    print(f"       image={imagepath}, gt={gt_landmark_path}")
                continue

            # NME compute
            try:
                nme = calculate_nme(
                    predicted_landmarks,
                    landmarks2dGT_2d,
                    normalization=args.normalization,
                    landmark_format='68'
                )
            except Exception as e:
                skipped_count += 1
                if args.verbose:
                    print(f"[Skip] NME calc failed: {e}")
                continue

            nme_list.append(nme)
            valid_count += 1

            # optional save kpts
            if args.saveKpt:
                # name이 None이면 파일명 기반으로
                if name is None:
                    base = os.path.splitext(os.path.basename(imagepath))[0] if imagepath else f"frame_{i:05d}"
                else:
                    base = name

                pred_save = os.path.join(savefolder, 'landmarks', f"{base}_pred.npy")
                gt_save = os.path.join(savefolder, 'landmarks', f"{base}_gt.npy")
                np.save(pred_save, pred_np)
                np.save(gt_save, landmarks2dGT_2d)

        # save results per dir
        if valid_count > 0:
            mean_nme = float(np.mean(nme_list))
            std_nme = float(np.std(nme_list))
            min_nme = float(np.min(nme_list))
            max_nme = float(np.max(nme_list))

            all_nme_results[rel_dir] = {
                'mean_nme': mean_nme,
                'std_nme': std_nme,
                'min_nme': min_nme,
                'max_nme': max_nme,
                'valid_count': int(valid_count),
                'skipped_count': int(skipped_count),
                'all_nme': nme_list,
                'normalization': args.normalization
            }

            print(f"\nNME Results for {rel_dir}:")
            print(f"  Mean NME: {mean_nme:.6f}")
            print(f"  Std NME:  {std_nme:.6f}")
            print(f"  Min NME:  {min_nme:.6f}")
            print(f"  Max NME:  {max_nme:.6f}")
            print(f"  Valid samples: {valid_count}")
            print(f"  Skipped samples: {skipped_count}")

            # json 저장
            result_json_file = os.path.join(savefolder, 'nme_results.json')
            with open(result_json_file, 'w') as f:
                json.dump(all_nme_results[rel_dir], f, indent=4)

            # txt 저장
            result_txt_file = os.path.join(savefolder, 'nme_results.txt')
            with open(result_txt_file, 'w') as f:
                f.write("NME (Normalized Mean Error) Calculation Results - AFEW-VA Dataset\n")
                f.write("=" * 60 + "\n")
                f.write(f"Directory: {rel_dir}\n")
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
            print(f"\nWarning: No valid samples found for {rel_dir}")

    # 전체 요약 저장
    if len(all_nme_results) > 0:
        # 전체 평균을 "디렉토리별 mean의 평균"으로 계산 (원본 동작 유지)
        all_means = [v['mean_nme'] for v in all_nme_results.values()]
        overall_mean = float(np.mean(all_means))
        overall_std = float(np.std(all_means))

        total_valid = int(sum(v['valid_count'] for v in all_nme_results.values()))
        total_skipped = int(sum(v['skipped_count'] for v in all_nme_results.values()))

        summary = {
            'overall_mean_nme': overall_mean,
            'overall_std_nme': overall_std,
            'total_valid': total_valid,
            'total_skipped': total_skipped,
            'num_dirs': len(all_nme_results),
            'normalization': args.normalization,
            'per_dir': all_nme_results
        }

        summary_file = os.path.join(base_savefolder, 'nme_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)

        print("\nAll directories processed.")
        print(f"Overall mean NME (dir-avg): {overall_mean:.6f}")
        print(f"Overall std NME (dir-avg):  {overall_std:.6f}")
        print(f"Total valid: {total_valid}")
        print(f"Total skipped: {total_skipped}")
        print(f"Results saved in {base_savefolder}")
    else:
        print("\nNo valid results found in any directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFEW-VA Landmark NME Evaluation Script')

    parser.add_argument('--inputpath',
                        type=str,
                        default='/home/cine/Downloads/AFEW-VA/croppedImages/',
                        help='Input path to AFEW-VA croppedImages root')

    parser.add_argument('--gt_landmark_dir',
                        type=str,
                        default='/home/cine/Downloads/AFEW-VA/lmkGT_N/*',
                        help='GT landmark path pattern, must include "*" at the end')

    parser.add_argument('--savefolder',
                        type=str,
                        default='/media/cine/First/HWPJ2/ProjectResult/Demos/NME_AFEW_Evaluation/DetailNew_20260103_mrf015/',
                        help='Folder to save results')

    parser.add_argument('--pretrained_modelpath_ViT',
                        type=str,
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_20260103_mrf015/model.tar',
                        help='Path to pretrained model (.tar)')

    # device default: CUDA_VISIBLE_DEVICES 고려(원본 흐름 최대한 유지)
    default_device = 'cuda:0'
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
            # 원본 코드에서 cuda:1로 쓰던 경우를 고려
            default_device = 'cuda:1'
    except Exception:
        pass

    parser.add_argument('--device',
                        type=str,
                        default=default_device,
                        help='Device to run on, e.g., cuda:0, cuda:1, cpu')

    parser.add_argument('--normalization',
                        type=str,
                        default='inter-ocular',
                        choices=['inter-ocular', 'inter-pupil', 'bbox', 'face_size'],
                        help='NME normalization method')

    parser.add_argument('--iscrop',
                        action='store_true',
                        help='Whether input images are uncropped and need cropping (usually False for croppedImages)')

    parser.add_argument('--rasterizer_type',
                        type=str,
                        default='pytorch3d',
                        help='Rasterizer type (pytorch3d)')

    parser.add_argument('--useTex',
                        action='store_true',
                        default=True,
                        help='Use texture model')

    parser.add_argument('--extractTex',
                        action='store_true',
                        default=False,
                        help='Extract texture')

    parser.add_argument('--saveKpt',
                        action='store_true',
                        default=False,
                        help='Save predicted/GT landmarks as .npy')

    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Verbose logging for skipped samples')

    main(parser.parse_args())

