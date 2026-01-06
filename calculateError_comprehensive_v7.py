"""
통합 평가 코드 v2: 랜드마크 NME와 픽셀 NME를 함께 계산 (개선 버전)
랜드마크 NME: 얼굴 랜드마크 위치 정확도 평가
픽셀 NME: 이미지 재구성 품질 평가 (픽셀별 차이)

개선 사항:
1. 정렬 보장: 이미지 크기 검증 및 리사이즈, 얼굴 위치/회전/스케일 정렬
2. 마스크 적용: 얼굴 영역 마스크 생성 및 적용
3. 버그 수정: 크기 불일치 처리, valid_pixels 계산 수정
4. Interocular 정의: 랜드마크에서 실제 interocular distance 계산 및 사용
"""
import os.path
import os
import argparse

from tensor_cropper import transform_points, batch_kp_2d_l1_loss
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm


def calculate_interocular_distance(landmarks):
    """
    FLAME 68 랜드마크에서 interocular distance 계산
    
    Args:
        landmarks: numpy array [68, 2] 또는 torch.Tensor [68, 2]
    
    Returns:
        interocular_dist: 눈 사이 거리 (스칼라)
    """
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    
    # FLAME 68 랜드마크 인덱스
    # 왼쪽 눈: 36, 37, 38, 39, 40, 41
    # 오른쪽 눈: 42, 43, 44, 45, 46, 47
    left_eye_indices = [36, 37, 38, 39, 40, 41]
    right_eye_indices = [42, 43, 44, 45, 46, 47]
    
    # 눈 중심점 계산
    left_eye_center = np.mean(landmarks[left_eye_indices, :2], axis=0)
    right_eye_center = np.mean(landmarks[right_eye_indices, :2], axis=0)
    
    # 눈 사이 거리 계산
    interocular_dist = np.linalg.norm(left_eye_center - right_eye_center)
    
    return interocular_dist


def create_face_mask_from_landmarks(landmarks, image_shape, dilation_factor=1.2):
    """
    랜드마크로부터 얼굴 영역 마스크 생성
    
    Args:
        landmarks: numpy array [68, 2] 또는 torch.Tensor [68, 2]
        image_shape: (H, W) 튜플
        dilation_factor: 얼굴 영역 확장 계수 (기본값: 1.2)
    
    Returns:
        mask: numpy array [H, W], 값 범위 [0, 1]
    """
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
    
    H, W = image_shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    
    # 얼굴 윤곽 랜드마크 인덱스 (FLAME 68: 0-16)
    face_contour_indices = list(range(17))
    
    # 얼굴 윤곽 점들
    face_points = landmarks[face_contour_indices, :2].astype(np.int32)
    
    # 얼굴 영역의 바운딩 박스 계산
    x_min = max(0, int(np.min(face_points[:, 0]) * (2 - dilation_factor)))
    x_max = min(W, int(np.max(face_points[:, 0]) * dilation_factor))
    y_min = max(0, int(np.min(face_points[:, 1]) * (2 - dilation_factor)))
    y_max = min(H, int(np.max(face_points[:, 1]) * dilation_factor))
    
    # 얼굴 윤곽으로 마스크 채우기
    if len(face_points) > 2:
        cv2.fillPoly(mask, [face_points], 1.0)
    
    # 얼굴 내부 영역도 포함 (얼굴 윤곽 + 눈, 코, 입 영역)
    # 얼굴 내부 랜드마크 인덱스 (17-67)
    inner_face_indices = list(range(17, 68))
    inner_points = landmarks[inner_face_indices, :2].astype(np.int32)
    
    # 얼굴 내부 영역을 convex hull로 채우기
    if len(inner_points) > 2:
        hull = cv2.convexHull(inner_points)
        cv2.fillPoly(mask, [hull], 1.0)
    
    # 바운딩 박스 내부를 마스크로 채우기 (간단한 방법)
    mask[y_min:y_max, x_min:x_max] = 1.0
    
    return mask


def align_images_by_landmarks(image_gt, image_pred, landmarks_gt, landmarks_pred, target_size=None):
    """
    랜드마크를 기반으로 이미지 정렬 (크기, 위치, 회전)
    
    Args:
        image_gt: GT 이미지 [H, W, 3]
        image_pred: 예측 이미지 [H, W, 3]
        landmarks_gt: GT 랜드마크 [68, 2]
        landmarks_pred: 예측 랜드마크 [68, 2]
        target_size: 정렬 후 목표 크기 (H, W), None이면 GT 이미지 크기 사용
    
    Returns:
        image_gt_aligned: 정렬된 GT 이미지
        image_pred_aligned: 정렬된 예측 이미지
        mask_gt: GT 이미지의 얼굴 마스크
        mask_pred: 예측 이미지의 얼굴 마스크
    """
    if isinstance(landmarks_gt, torch.Tensor):
        landmarks_gt = landmarks_gt.cpu().numpy()
    if isinstance(landmarks_pred, torch.Tensor):
        landmarks_pred = landmarks_pred.cpu().numpy()
    
    H_gt, W_gt = image_gt.shape[:2]
    H_pred, W_pred = image_pred.shape[:2]
    
    # 목표 크기 설정
    if target_size is None:
        target_size = (H_gt, W_gt)
    target_H, target_W = target_size
    
    # 1. 크기 정렬: 두 이미지를 동일한 크기로 리사이즈
    if (H_gt, W_gt) != (target_H, target_W):
        image_gt = cv2.resize(image_gt, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        landmarks_gt = landmarks_gt * np.array([target_W / W_gt, target_H / H_gt])
    
    if (H_pred, W_pred) != (target_H, target_W):
        image_pred = cv2.resize(image_pred, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        landmarks_pred = landmarks_pred * np.array([target_W / W_pred, target_H / H_pred])
    
    # 2. 위치 정렬: 얼굴 중심을 이미지 중심으로 이동
    # 얼굴 중심 계산 (모든 랜드마크의 평균)
    face_center_gt = np.mean(landmarks_gt[:, :2], axis=0)
    face_center_pred = np.mean(landmarks_pred[:, :2], axis=0)
    image_center = np.array([target_W / 2, target_H / 2])
    
    # 이동 벡터 계산
    translation_gt = image_center - face_center_gt
    translation_pred = image_center - face_center_pred
    
    # 이미지와 랜드마크 이동
    M_gt = np.float32([[1, 0, translation_gt[0]], [0, 1, translation_gt[1]]])
    M_pred = np.float32([[1, 0, translation_pred[0]], [0, 1, translation_pred[1]]])
    
    image_gt_aligned = cv2.warpAffine(image_gt, M_gt, (target_W, target_H), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    image_pred_aligned = cv2.warpAffine(image_pred, M_pred, (target_W, target_H),
                                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    landmarks_gt_aligned = landmarks_gt[:, :2] + translation_gt
    landmarks_pred_aligned = landmarks_pred[:, :2] + translation_pred
    
    # 3. 스케일 정렬: interocular distance를 기준으로 스케일 조정
    ioc_gt = calculate_interocular_distance(landmarks_gt_aligned)
    ioc_pred = calculate_interocular_distance(landmarks_pred_aligned)
    
    if ioc_gt > 0 and ioc_pred > 0:
        scale_factor = ioc_gt / ioc_pred
        
        # 예측 이미지 스케일 조정
        M_scale = cv2.getRotationMatrix2D(tuple(image_center), 0, scale_factor)
        image_pred_aligned = cv2.warpAffine(image_pred_aligned, M_scale, (target_W, target_H),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        landmarks_pred_aligned = (landmarks_pred_aligned - image_center) * scale_factor + image_center
    
    # 4. 회전 정렬: 눈 위치를 기준으로 회전 조정 (선택적, 간단한 버전)
    # 왼쪽 눈 중심과 오른쪽 눈 중심의 각도 차이 계산
    left_eye_gt = np.mean(landmarks_gt_aligned[[36, 37, 38, 39, 40, 41], :2], axis=0)
    right_eye_gt = np.mean(landmarks_gt_aligned[[42, 43, 44, 45, 46, 47], :2], axis=0)
    left_eye_pred = np.mean(landmarks_pred_aligned[[36, 37, 38, 39, 40, 41], :2], axis=0)
    right_eye_pred = np.mean(landmarks_pred_aligned[[42, 43, 44, 45, 46, 47], :2], axis=0)
    
    eye_vector_gt = right_eye_gt - left_eye_gt
    eye_vector_pred = right_eye_pred - left_eye_pred
    
    angle_gt = np.arctan2(eye_vector_gt[1], eye_vector_gt[0])
    angle_pred = np.arctan2(eye_vector_pred[1], eye_vector_pred[0])
    angle_diff = np.degrees(angle_gt - angle_pred)
    
    # 회전 조정 (작은 각도만, 너무 크면 오히려 왜곡될 수 있음)
    if abs(angle_diff) < 15:  # 15도 이내만 조정
        M_rotate = cv2.getRotationMatrix2D(tuple(image_center), angle_diff, 1.0)
        image_pred_aligned = cv2.warpAffine(image_pred_aligned, M_rotate, (target_W, target_H),
                                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # 마스크 생성 (정렬된 랜드마크 사용)
    mask_gt = create_face_mask_from_landmarks(landmarks_gt_aligned, (target_H, target_W))
    mask_pred = create_face_mask_from_landmarks(landmarks_pred_aligned, (target_H, target_W))
    
    # 두 마스크의 교집합 사용 (공통 얼굴 영역만)
    mask_combined = np.minimum(mask_gt, mask_pred)
    
    return image_gt_aligned, image_pred_aligned, mask_combined


def calculate_landmark_nme(landmarks_gt, landmarks_pred):
    """
    랜드마크 NME (Normalized Mean Error) 계산
    
    Args:
        landmarks_gt: Ground truth 랜드마크 [68, 2] 또는 [68, 3]
        landmarks_pred: 예측된 랜드마크 [68, 2]
    
    Returns:
        nme: 정규화된 평균 에러 (스칼라)
    """
    # GT가 3D인 경우 2D만 사용
    if landmarks_gt.shape[1] >= 2:
        landmarks_gt_2d = landmarks_gt[:, :2] if landmarks_gt.shape[1] > 2 else landmarks_gt
    else:
        return None
    
    # batch_kp_2d_l1_loss 사용 (기존 방식과 동일)
    # 입력 형식: [1, 68, 3] (마지막 차원은 visibility)
    landmarks_gt_with_vis = torch.cat([landmarks_gt_2d, torch.ones(landmarks_gt_2d.shape[0], 1)], dim=1)
    landmarks_gt_with_vis = landmarks_gt_with_vis[None, ...]  # [1, 68, 3]
    
    nme = batch_kp_2d_l1_loss(landmarks_gt_with_vis, landmarks_pred[None, ...])
    return nme.item()


def calculate_pixel_nme(image_gt, image_pred, face_mask=None, normalization='face_size', interocular_dist=None):
    """
    픽셀 NME (Normalized Mean Error) 계산 (개선 버전)
    
    Args:
        image_gt: Ground truth 이미지 [H, W, 3] 또는 [3, H, W], 값 범위 [0, 1] 또는 [0, 255]
        image_pred: 예측된 이미지 [H, W, 3] 또는 [3, H, W], 값 범위 [0, 1] 또는 [0, 255]
        face_mask: 얼굴 영역 마스크 [H, W] (선택적, None이면 전체 이미지 사용)
        normalization: 정규화 방법 ('face_size', 'image_size', 'interocular')
        interocular_dist: interocular distance (normalization='interocular'일 때 필요)
    
    Returns:
        nme: 정규화된 평균 픽셀 에러 (스칼라)
    """
    # 이미지 형식 통일: [H, W, 3]
    if len(image_gt.shape) == 3 and image_gt.shape[0] == 3:
        # [3, H, W] -> [H, W, 3]
        image_gt = image_gt.transpose(1, 2, 0)
    if len(image_pred.shape) == 3 and image_pred.shape[0] == 3:
        image_pred = image_pred.transpose(1, 2, 0)
    
    # 버그 수정 1: 이미지 크기 검증 및 리사이즈
    H_gt, W_gt = image_gt.shape[:2]
    H_pred, W_pred = image_pred.shape[:2]
    
    if (H_gt, W_gt) != (H_pred, W_pred):
        # 예측 이미지를 GT 이미지 크기로 리사이즈
        image_pred = cv2.resize(image_pred, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
        H_pred, W_pred = H_gt, W_gt
        
        # 마스크도 리사이즈
        if face_mask is not None:
            face_mask = cv2.resize(face_mask, (W_gt, H_gt), interpolation=cv2.INTER_NEAREST)
    
    # 값 범위 통일: [0, 1]
    if image_gt.max() > 1.0:
        image_gt = image_gt / 255.0
    if image_pred.max() > 1.0:
        image_pred = image_pred / 255.0
    
    H, W = image_gt.shape[:2]
    
    # 픽셀별 절대 차이 계산
    pixel_diff = np.abs(image_gt.astype(np.float32) - image_pred.astype(np.float32))
    
    # 마스크 적용 (얼굴 영역만 사용)
    if face_mask is not None:
        # 버그 수정 2: 마스크 크기 검증
        if face_mask.shape[:2] != (H, W):
            face_mask = cv2.resize(face_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if len(face_mask.shape) == 3:
            face_mask = face_mask[:, :, 0]  # [H, W, 1] -> [H, W]
        face_mask = face_mask.astype(np.float32)
        
        # 마스크를 RGB 채널에 맞게 확장
        face_mask_3d = face_mask[..., None]  # [H, W, 1]
        pixel_diff = pixel_diff * face_mask_3d
        valid_pixels = np.sum(face_mask)
    else:
        valid_pixels = H * W
    
    # 정규화 방법에 따라 normalization factor 결정
    if normalization == 'face_size':
        # 얼굴 영역 크기로 정규화 (마스크가 있는 경우)
        if face_mask is not None:
            normalization_factor = valid_pixels
        else:
            normalization_factor = H * W
    elif normalization == 'image_size':
        # 이미지 크기로 정규화
        normalization_factor = H * W
    elif normalization == 'interocular':
        # Interocular 정의 정리: 실제 interocular distance 사용
        if interocular_dist is not None and interocular_dist > 0:
            normalization_factor = interocular_dist ** 2  # 면적 단위로 정규화
        else:
            # interocular distance가 없으면 이미지 크기 사용 (fallback)
            print("Warning: interocular distance not provided, using image_size as fallback")
            normalization_factor = H * W
    else:
        normalization_factor = H * W
    
    # NME 계산: 모든 픽셀의 차이 합 / (정규화 인자 * 채널 수)
    total_error = np.sum(pixel_diff)
    nme = total_error / (normalization_factor * 3.0 + 1e-8)  # 3은 RGB 채널
    
    return nme


def load_image(image_path):
    """
    이미지 로드
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        image: numpy array [H, W, 3], 값 범위 [0, 1]
    """
    if not os.path.exists(image_path):
        return None
    
    # OpenCV로 이미지 로드 (BGR 형식)
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # BGR -> RGB 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # [0, 255] -> [0, 1] 정규화
    image = image.astype(np.float32) / 255.0
    
    return image


def main(args):
    """
    메인 함수: 랜드마크 NME와 픽셀 NME를 함께 계산 (개선 버전)
    """
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
    
    # GT 경로 설정
    landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"
    # GT 이미지 경로 (원본 데이터셋 경로, 필요시 수정)
    imageDir_GT = "/home/cine/Downloads/AFEW-VA/croppedImages/*"
    
    # 에러 누적 변수 (랜드마크 NME)
    landmark_coarse = 0.  # Coarse mesh에서 계산한 랜드마크
    landmark_detail = 0.  # Detail 이미지에서 검출한 랜드마크
    landmark_deca = 0.    # DECA baseline
    
    # 에러 누적 변수 (픽셀 NME)
    pixel_coarse = 0.     # Coarse 재구성 이미지
    pixel_detail = 0.     # Detail 재구성 이미지
    pixel_deca = 0.       # DECA 재구성 이미지
    
    # 통계 변수
    length = len([p for p in landmarkDir if "_DECA" not in p and "_detail" not in p])
    
    # 랜드마크 NME 통계
    max_landmark_coarse = 0
    min_landmark_coarse = 1000
    max_landmark_detail = 0
    min_landmark_detail = 1000
    max_landmark_deca = 0
    min_landmark_deca = 1000
    
    # 픽셀 NME 통계
    max_pixel_coarse = 0
    min_pixel_coarse = 1000
    max_pixel_detail = 0
    min_pixel_detail = 1000
    max_pixel_deca = 0
    min_pixel_deca = 1000
    
    # 각 샘플별 결과 저장
    sample_results = []
    
    for landmarkpath in tqdm(landmarkDir, desc="Evaluating"):
        # DECA나 detail 파일은 건너뛰기
        if "DECA" in landmarkpath or "detail" in landmarkpath:
            continue
        
        try:
            name = os.path.splitext(os.path.split(landmarkpath)[-1])[0]
            
            # ========== 랜드마크 NME 계산 ==========
            if args.calc_landmark:
                # Coarse mesh에서 계산한 랜드마크
                landmarks2d_coarse = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))
                
                # DECA baseline 랜드마크
                landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace(".npy", "_DECA.npy"), allow_pickle=True))
                
                # GT 랜드마크 먼저 로드 (다른 계산에도 필요)
                if f"{model_dir}/" in landmarkpath:
                    path_after_model_dir = landmarkpath.split(f"{model_dir}/")[-1]
                elif "AFWE_VA/" in landmarkpath:
                    path_after_model_dir = landmarkpath.split("AFWE_VA/")[-1]
                else:
                    parts = landmarkpath.split("/")
                    path_after_model_dir = "/".join(parts[-3:])
                
                gt_relative_path = path_after_model_dir.replace("2d_landmark_68/", "")
                landmarkpathGT = landmarkDir_GT.replace("*", gt_relative_path)
                
                if not os.path.exists(landmarkpathGT):
                    print(f"Warning: GT landmark not found for {name}, skipping...")
                    length -= 1
                    continue
                
                landmarks2dGT = torch.from_numpy(np.load(landmarkpathGT, allow_pickle=True))
                
                if landmarks2dGT.shape[1] >= 2:
                    landmarks2dGT_2d = landmarks2dGT[:, :2] if landmarks2dGT.shape[1] > 2 else landmarks2dGT
                else:
                    print(f"Warning: Invalid GT landmark shape for {name}, skipping...")
                    length -= 1
                    continue
                
                # 랜드마크 NME 계산 (Coarse, DECA)
                landmark_coarse_val = calculate_landmark_nme(landmarks2dGT_2d, landmarks2d_coarse)
                landmark_deca_val = calculate_landmark_nme(landmarks2dGT_2d, landmarks2d_DECA)
                
                # Detail 이미지에서 검출한 랜드마크
                detail_landmark_path = landmarkpath.replace(".npy", "_detail.npy")
                landmark_detail_val = None
                if os.path.exists(detail_landmark_path):
                    landmarks2d_detail_68 = np.load(detail_landmark_path, allow_pickle=True)
                    if landmarks2d_detail_68.shape[0] == 68:
                        landmarks2d_detail = torch.from_numpy(landmarks2d_detail_68).float()
                        landmark_detail_val = calculate_landmark_nme(landmarks2dGT_2d, landmarks2d_detail)
                    else:
                        print(f"Warning: Expected 68 landmarks, got {landmarks2d_detail_68.shape[0]} for {name}")
                else:
                    print(f"Warning: Detail landmark not found for {name}, skipping detail landmark evaluation...")
                
                # 누적 및 통계 업데이트
                if landmark_coarse_val is not None:
                    landmark_coarse += landmark_coarse_val
                    if landmark_coarse_val < min_landmark_coarse:
                        min_landmark_coarse = landmark_coarse_val
                    elif landmark_coarse_val > max_landmark_coarse:
                        max_landmark_coarse = landmark_coarse_val
                
                if landmark_deca_val is not None:
                    landmark_deca += landmark_deca_val
                    if landmark_deca_val < min_landmark_deca:
                        min_landmark_deca = landmark_deca_val
                    elif landmark_deca_val > max_landmark_deca:
                        max_landmark_deca = landmark_deca_val
                
                if landmark_detail_val is not None:
                    landmark_detail += landmark_detail_val
                    if landmark_detail_val < min_landmark_detail:
                        min_landmark_detail = landmark_detail_val
                    elif landmark_detail_val > max_landmark_detail:
                        max_landmark_detail = landmark_detail_val
            else:
                landmark_coarse_val = None
                landmark_detail_val = None
                landmark_deca_val = None
                landmarks2dGT_2d = None
                landmarks2d_coarse = None
                landmarks2d_DECA = None
            
            # ========== 픽셀 NME 계산 (개선 버전) ==========
            if args.calc_pixel:
                # 재구성된 이미지 경로 찾기
                # 예: .../2d_landmark_68/00001.npy -> .../result/00001.jpg
                result_dir = os.path.dirname(os.path.dirname(landmarkpath))
                result_image_path = os.path.join(result_dir, 'result', name + '.jpg')
                
                # GT 이미지 경로 찾기
                if f"{model_dir}/" in landmarkpath:
                    path_after_model_dir = landmarkpath.split(f"{model_dir}/")[-1]
                elif "AFWE_VA/" in landmarkpath:
                    path_after_model_dir = landmarkpath.split("AFWE_VA/")[-1]
                else:
                    parts = landmarkpath.split("/")
                    path_after_model_dir = "/".join(parts[-3:])
                
                # GT 이미지 경로 구성 (예: 01/001/00001.jpg)
                gt_image_relative_path = path_after_model_dir.replace("2d_landmark_68/", "").replace(".npy", ".jpg")
                imagepathGT = imageDir_GT.replace("*", gt_image_relative_path)
                
                # 이미지 로드
                image_gt = load_image(imagepathGT)
                image_pred = load_image(result_image_path)
                
                if image_gt is None or image_pred is None:
                    if args.calc_pixel:
                        print(f"Warning: Image not found for {name}, skipping pixel evaluation...")
                        pixel_coarse_val = None
                        pixel_detail_val = None
                        pixel_deca_val = None
                    else:
                        pixel_coarse_val = None
                        pixel_detail_val = None
                        pixel_deca_val = None
                else:
                    # 개선 사항 1: 정렬 보장
                    # 랜드마크가 있는 경우 정렬 수행
                    if args.calc_landmark and landmarks2dGT_2d is not None and landmarks2d_coarse is not None:
                        try:
                            image_gt_aligned, image_pred_aligned, face_mask = align_images_by_landmarks(
                                image_gt, image_pred, 
                                landmarks2dGT_2d, landmarks2d_coarse
                            )
                            
                            # Interocular distance 계산 (정렬 후)
                            interocular_dist = calculate_interocular_distance(landmarks2dGT_2d)
                            
                            # 개선 사항 2: 마스크 적용 및 개선 사항 4: Interocular 정의
                            pixel_coarse_val = calculate_pixel_nme(
                                image_gt_aligned, image_pred_aligned, 
                                face_mask=face_mask,
                                normalization=args.pixel_normalization,
                                interocular_dist=interocular_dist if args.pixel_normalization == 'interocular' else None
                            )
                        except Exception as e:
                            print(f"Warning: Alignment failed for {name}: {e}, using original images without alignment")
                            # 정렬 실패 시 원본 이미지 사용 (마스크 없이)
                            interocular_dist = calculate_interocular_distance(landmarks2dGT_2d) if landmarks2dGT_2d is not None else None
                            pixel_coarse_val = calculate_pixel_nme(
                                image_gt, image_pred,
                                face_mask=None,  # 정렬 실패 시 마스크 없이
                                normalization=args.pixel_normalization,
                                interocular_dist=interocular_dist if args.pixel_normalization == 'interocular' else None
                            )
                    else:
                        # 랜드마크가 없는 경우 원본 이미지 사용 (마스크 없이)
                        interocular_dist = None
                        pixel_coarse_val = calculate_pixel_nme(
                            image_gt, image_pred,
                            face_mask=None,
                            normalization=args.pixel_normalization,
                            interocular_dist=None
                        )
                    
                    # 현재는 하나의 재구성 이미지만 있으므로 동일한 이미지 사용
                    # 향후 coarse/detail/deca 이미지를 구분할 수 있으면 분리 가능
                    pixel_detail_val = pixel_coarse_val  # 동일 이미지 사용
                    pixel_deca_val = pixel_coarse_val    # 동일 이미지 사용
                    
                    # 누적 및 통계 업데이트
                    if pixel_coarse_val is not None:
                        pixel_coarse += pixel_coarse_val
                        pixel_detail += pixel_detail_val
                        pixel_deca += pixel_deca_val
                        
                        if pixel_coarse_val < min_pixel_coarse:
                            min_pixel_coarse = pixel_coarse_val
                        elif pixel_coarse_val > max_pixel_coarse:
                            max_pixel_coarse = pixel_coarse_val
                        
                        if pixel_detail_val < min_pixel_detail:
                            min_pixel_detail = pixel_detail_val
                        elif pixel_detail_val > max_pixel_detail:
                            max_pixel_detail = pixel_detail_val
                        
                        if pixel_deca_val < min_pixel_deca:
                            min_pixel_deca = pixel_deca_val
                        elif pixel_deca_val > max_pixel_deca:
                            max_pixel_deca = pixel_deca_val
            else:
                pixel_coarse_val = None
                pixel_detail_val = None
                pixel_deca_val = None
            
            # 샘플별 결과 저장
            sample_results.append({
                'name': name,
                'landmark_coarse': landmark_coarse_val,
                'landmark_detail': landmark_detail_val,
                'landmark_deca': landmark_deca_val,
                'pixel_coarse': pixel_coarse_val,
                'pixel_detail': pixel_detail_val,
                'pixel_deca': pixel_deca_val
            })
                
        except Exception as e:
            print(f"Error processing {landmarkpath}: {e}")
            import traceback
            traceback.print_exc()
            length -= 1
            continue
    
    # ========== 결과 출력 및 저장 ==========
    results = []
    
    # 랜드마크 NME 결과
    if args.calc_landmark:
        results.append("=" * 70)
        results.append("랜드마크 NME (Normalized Mean Error)")
        results.append("=" * 70)
        results.append("DECA (baseline): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f}".format(
            landmark_deca, landmark_deca/length if length > 0 else 0, max_landmark_deca, min_landmark_deca))
        results.append("Coarse (coarse mesh): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f}".format(
            landmark_coarse, landmark_coarse/length if length > 0 else 0, max_landmark_coarse, min_landmark_coarse))
        # Detail 결과가 있는 경우에만 출력
        detail_count = sum(1 for r in sample_results if r['landmark_detail'] is not None)
        if detail_count > 0:
            results.append("Detail (detail image): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f} (samples: {})".format(
                landmark_detail, landmark_detail/detail_count if detail_count > 0 else 0, 
                max_landmark_detail, min_landmark_detail, detail_count))
    
    # 픽셀 NME 결과
    if args.calc_pixel:
        results.append("")
        results.append("=" * 70)
        results.append("픽셀 NME (Normalized Mean Error) - 개선 버전")
        results.append("=" * 70)
        results.append("정규화 방법: {}".format(args.pixel_normalization))
        results.append("개선 사항: 정렬 보장, 마스크 적용, 버그 수정, Interocular 정의 정리")
        results.append("DECA (baseline): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f}".format(
            pixel_deca, pixel_deca/length if length > 0 else 0, max_pixel_deca, min_pixel_deca))
        results.append("Coarse (coarse mesh): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f}".format(
            pixel_coarse, pixel_coarse/length if length > 0 else 0, max_pixel_coarse, min_pixel_coarse))
        # Detail 결과가 있는 경우에만 출력
        pixel_detail_count = sum(1 for r in sample_results if r['pixel_detail'] is not None)
        if pixel_detail_count > 0:
            results.append("Detail (detail image): Total={:.6f}, Avg={:.6f}, Max={:.6f}, Min={:.6f} (samples: {})".format(
                pixel_detail, pixel_detail/pixel_detail_count if pixel_detail_count > 0 else 0,
                max_pixel_detail, min_pixel_detail, pixel_detail_count))
    
    # 콘솔 출력
    for result in results:
        print(result)
    
    # 파일로 저장
    output_file = os.path.join(output_dir, f"comprehensive_error_results_{model_dir}_v2.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("통합 평가 결과 (랜드마크 NME + 픽셀 NME) - 개선 버전 v2\n")
        f.write("=" * 70 + "\n")
        f.write("모델 디렉토리: {}\n".format(model_dir))
        f.write("총 샘플 수: {}\n".format(length))
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            f.write(result + "\n")
        
        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("참고사항\n")
        f.write("=" * 70 + "\n")
        f.write("- 랜드마크 NME: 얼굴 랜드마크 위치 정확도 평가\n")
        f.write("- 픽셀 NME: 이미지 재구성 품질 평가 (픽셀별 차이)\n")
        f.write("- 정규화 방법: {}\n".format(args.pixel_normalization if args.calc_pixel else "N/A"))
        f.write("\n")
        f.write("개선 사항:\n")
        f.write("1. 정렬 보장: 이미지 크기 검증 및 리사이즈, 얼굴 위치/회전/스케일 정렬\n")
        f.write("2. 마스크 적용: 얼굴 영역 마스크 생성 및 적용\n")
        f.write("3. 버그 수정: 크기 불일치 처리, valid_pixels 계산 수정\n")
        f.write("4. Interocular 정의: 랜드마크에서 실제 interocular distance 계산 및 사용\n")
        f.write("\n")
        f.write("Format: Model: Total_Error Average_Error Max_Error Min_Error\n")
    
    print(f"\n결과가 저장되었습니다: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='통합 평가 v2: 랜드마크 NME와 픽셀 NME를 함께 계산 (개선 버전)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python calculateError_comprehensive_v2.py --model_path /path/to/model.tar
  python calculateError_comprehensive_v2.py --model_path /path/to/model.tar --calc_landmark --calc_pixel
  python calculateError_comprehensive_v2.py --model_path /path/to/model.tar --calc_pixel --pixel_normalization interocular
        """
    )
    
    parser.add_argument('--model_path', 
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune/model.tar',
                        type=str,
                        help='모델 파일 경로 (.tar). 부모 디렉토리 이름이 결과 디렉토리 이름으로 사용됩니다.')
    
    parser.add_argument('--calc_landmark', 
                        action='store_true',
                        default=True,
                        help='랜드마크 NME 계산 여부 (기본값: True, 픽셀 정렬에 필요)')
    
    parser.add_argument('--calc_pixel', 
                        action='store_true',
                        default=True,
                        help='픽셀 NME 계산 여부 (기본값: True)')
    
    parser.add_argument('--pixel_normalization',
                        type=str,
                        default='face_size',
                        choices=['face_size', 'image_size', 'interocular'],
                        help='픽셀 NME 정규화 방법 (기본값: face_size)')
    
    args = parser.parse_args()
    
    # 기본값 처리: 둘 다 False면 둘 다 True로 설정
    if not args.calc_landmark and not args.calc_pixel:
        args.calc_landmark = True
        args.calc_pixel = True
        print("Warning: --calc_landmark와 --calc_pixel이 모두 False입니다. 둘 다 True로 설정합니다.")
    
    # 픽셀 NME 계산 시 랜드마크가 필요함 (정렬 및 마스크 생성용)
    if args.calc_pixel and not args.calc_landmark:
        print("Warning: --calc_pixel requires --calc_landmark for image alignment and mask generation.")
        print("Setting --calc_landmark=True automatically.")
        args.calc_landmark = True
    
    main(args)

