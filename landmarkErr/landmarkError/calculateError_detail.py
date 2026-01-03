import os.path
import os
import argparse

from tensor_cropper import transform_points, batch_kp_2d_l1_loss
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm

# MediaPipe 468개 landmark를 68개 FLAME landmark로 매핑하는 인덱스
# 출처: MediaPipe Face Mesh와 FLAME 68 landmark 매핑
# 주의: 이 매핑은 근사치이며, 정확한 매핑을 위해서는 별도의 검증이 필요할 수 있습니다.
MEDIAPIPE_TO_68_INDICES = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,  # 얼굴 윤곽 (17개)
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,  # 왼쪽 눈썹 (10개)
    168, 197, 5, 4, 75, 97, 2, 326, 305,  # 코 (9개)
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # 오른쪽 눈 (12개)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87  # 입 (20개)
]


def map_mediapipe_to_68(mediapipe_landmarks):
    """
    MediaPipe 468개 landmark를 68개 FLAME landmark로 매핑
    
    Args:
        mediapipe_landmarks: numpy array, shape [468, 2] 또는 [468, 3]
    
    Returns:
        landmarks_68: numpy array, shape [68, 2]
    """
    if mediapipe_landmarks is None:
        return None
    
    if mediapipe_landmarks.shape[0] != 468:
        print(f"Warning: Expected 468 landmarks, got {mediapipe_landmarks.shape[0]}")
        return None
    
    # 인덱스로 선택 (2D만 사용)
    if mediapipe_landmarks.shape[1] >= 2:
        landmarks_68 = mediapipe_landmarks[MEDIAPIPE_TO_68_INDICES, :2]  # [68, 2]
    else:
        landmarks_68 = mediapipe_landmarks[MEDIAPIPE_TO_68_INDICES]  # [68]
        landmarks_68 = landmarks_68.reshape(-1, 1)  # [68, 1] -> [68, 2]로 확장 필요
    
    return landmarks_68


def main(args):
    # 모델 경로에서 디렉토리 이름 추출
    model_path = args.model_path
    model_dir = os.path.basename(os.path.dirname(model_path))
    
    # 입력 경로 자동 생성
    landmark_input_pattern = f"/media/cine/First/HWPJ2/ProjectResult/AFWE_VA/{model_dir}/*/*/2d_landmark_68/*.npy"
    landmarkDir = sorted(glob(landmark_input_pattern))
    
    if len(landmarkDir) == 0:
        print(f"Warning: No landmark files found in {landmark_input_pattern}")
        print("Please check if the model directory name matches the output directory.")
        return
    
    print(f"Found {len([p for p in landmarkDir if '_DECA' not in p and '_detail' not in p])} landmark files")
    
    # 출력 디렉토리 자동 생성
    output_dir = f"/media/cine/First/HWPJ2/ProjectResult/AFWE_landmarkErr/{model_dir}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"

    # 에러 누적 변수
    Coarse = 0.  # Coarse mesh에서 계산한 랜드마크
    Detail = 0.  # Detail 이미지에서 검출한 랜드마크
    DECA = 0.    # DECA baseline

    length = len([p for p in landmarkDir if "_DECA" not in p and "_detail" not in p])
    max_coarse = 0
    min_coarse = 1000
    max_detail = 0
    min_detail = 1000
    max_deca = 0
    min_deca = 1000

    for landmarkpath in tqdm(landmarkDir):
        # DECA나 detail 파일은 건너뛰기
        if "DECA" in landmarkpath or "detail" in landmarkpath:
            continue
        
        try:
            # Coarse mesh에서 계산한 랜드마크
            landmarks2d_coarse = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))
            name = os.path.splitext(os.path.split(landmarkpath)[-1])[0]
            
            # DECA baseline 랜드마크
            landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace(".npy", "_DECA.npy"), allow_pickle=True))
            
            # Detail 이미지에서 검출한 랜드마크 (이미 68개로 매핑됨)
            detail_landmark_path = landmarkpath.replace(".npy", "_detail.npy")
            if not os.path.exists(detail_landmark_path):
                print(f"Warning: Detail landmark not found for {name}, skipping...")
                length -= 1
                continue
            
            # 이미 68개로 매핑된 랜드마크를 로드
            landmarks2d_detail_68 = np.load(detail_landmark_path, allow_pickle=True)  # [68, 2]
            
            if landmarks2d_detail_68.shape[0] != 68:
                print(f"Warning: Expected 68 landmarks, got {landmarks2d_detail_68.shape[0]} for {name}, skipping...")
                length -= 1
                continue
            
            landmarks2d_detail = torch.from_numpy(landmarks2d_detail_68).float()
            
            # GT 랜드마크
            # 경로에서 모델 디렉토리명 이후 부분 추출 (예: 01/001/2d_landmark_68/00001.npy -> 01/001/00001.npy)
            if f"{model_dir}/" in landmarkpath:
                path_after_model_dir = landmarkpath.split(f"{model_dir}/")[-1]
            elif "AFWE_VA/" in landmarkpath:
                path_after_model_dir = landmarkpath.split("AFWE_VA/")[-1]
            else:
                # fallback: 마지막 3개 디렉토리 경로 사용
                parts = landmarkpath.split("/")
                path_after_model_dir = "/".join(parts[-3:])
            
            # 2d_landmark_68/ 제거하고 파일명만 남기기
            gt_relative_path = path_after_model_dir.replace("2d_landmark_68/", "")
            landmarkpathGT = landmarkDir_GT.replace("*", gt_relative_path)
            if not os.path.exists(landmarkpathGT):
                print(f"Warning: GT landmark not found for {name}, skipping...")
                length -= 1
                continue
            
            landmarks2dGT = torch.from_numpy(np.load(landmarkpathGT, allow_pickle=True))
            
            # GT가 3D인 경우 2D만 사용
            if landmarks2dGT.shape[1] >= 2:
                landmarks2dGT_2d = landmarks2dGT[:, :2] if landmarks2dGT.shape[1] > 2 else landmarks2dGT
            else:
                print(f"Warning: Invalid GT landmark shape for {name}, skipping...")
                length -= 1
                continue
            
            # 에러 계산
            loss_coarse = batch_kp_2d_l1_loss(landmarks2dGT_2d[None, ...], landmarks2d_coarse)
            loss_detail = batch_kp_2d_l1_loss(landmarks2dGT_2d[None, ...], landmarks2d_detail)
            loss_deca = batch_kp_2d_l1_loss(landmarks2dGT_2d[None, ...], landmarks2d_DECA)
            
            Coarse += loss_coarse
            Detail += loss_detail
            DECA += loss_deca
            
            # 최대/최소 업데이트
            if loss_coarse < min_coarse:
                min_coarse = loss_coarse
            elif loss_coarse > max_coarse:
                max_coarse = loss_coarse
            
            if loss_detail < min_detail:
                min_detail = loss_detail
            elif loss_detail > max_detail:
                max_detail = loss_detail
            
            if loss_deca < min_deca:
                min_deca = loss_deca
            elif loss_deca > max_deca:
                max_deca = loss_deca
                
        except Exception as e:
            print(f"Error processing {landmarkpath}: {e}")
            length -= 1
            continue

    # 결과 출력 및 저장
    results = []
    results.append("DECA: {} {} {} {}".format(DECA, DECA/length, max_deca, min_deca))
    results.append("Coarse: {} {} {} {}".format(Coarse, Coarse/length, max_coarse, min_coarse))
    results.append("Detail: {} {} {} {}".format(Detail, Detail/length, max_detail, min_detail))

    # 콘솔 출력
    for result in results:
        print(result)

    # 파일로 저장 (모델 디렉토리 이름 포함)
    output_file = os.path.join(output_dir, f"landmark_error_results_detail_{model_dir}.txt")
    with open(output_file, 'w') as f:
        f.write("Landmark Error Calculation Results (Detail vs Coarse)\n")
        f.write("=" * 50 + "\n")
        f.write("Total samples: {}\n".format(length))
        f.write("=" * 50 + "\n")
        f.write("DECA (baseline): Total_Error Average_Error Max_Error Min_Error\n")
        f.write("Coarse (coarse mesh): Total_Error Average_Error Max_Error Min_Error\n")
        f.write("Detail (detail image): Total_Error Average_Error Max_Error Min_Error\n")
        f.write("=" * 50 + "\n")
        for result in results:
            f.write(result + "\n")
        f.write("\n")
        f.write("Format: Model: Total_Error Average_Error Max_Error Min_Error\n")
        f.write("\n")
        f.write("Note: Detail landmarks are detected from rendered detail images using MediaPipe\n")
        f.write("      and mapped from 468 MediaPipe landmarks to 68 FLAME landmarks.\n")

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate landmark error for detail vs coarse mesh')
    parser.add_argument('--model_path', 
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune/model.tar',
                        type=str,
                        help='Path to the model.tar file. The parent directory name will be used to find landmark files.')
    args = parser.parse_args()
    main(args)

