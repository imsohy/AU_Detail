"""
2025-12-20 23:00작성.
이전 버전: demo_testAU_OnlyE_ForAFWE_WJ.py
번경 이유:
detail로 렌더링된 이미지에 랜드마크 디텍션을 해서 468개의 mediapipe landmark를 만들고 이를
68개 FLAME landmark로 매핑하여 저장하는 기능을 추가한 파일이다. 근데 잘 아됐다
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
import glob
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

# import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.gatfarec_Video_OnlyExpress_WT_DetailNew_WJ import DECA
from decalib.datasets import datasets_WT_DetailNew as datasets
from decalib.utils import util
from decalib.utils.config_wt_DetailNew import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points

# MediaPipe landmark detector 추가
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# MediaPipe 468개 landmark를 68개 FLAME landmark로 매핑하는 인덱스
MEDIAPIPE_TO_68_INDICES = [
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,  # 얼굴 윤곽 (17개)
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,  # 왼쪽 눈썹 (10개)
    168, 197, 5, 4, 75, 97, 2, 326, 305,  # 코 (9개)
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,  # 오른쪽 눈 (12개)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181, 78, 82, 13, 312, 308, 317, 14, 87  # 입 (20개)
]


def map_mediapipe_to_68(mediapipe_landmarks):
    """
    MediaPipe 468개 또는 478개 landmark를 68개 FLAME landmark로 매핑
    (refine_landmarks=True일 때 478개가 반환됨)
    
    Args:
        mediapipe_landmarks: numpy array, shape [468, 2] 또는 [478, 2] 또는 [468, 3] 또는 [478, 3]
    
    Returns:
        landmarks_68: numpy array, shape [68, 2]
    """
    if mediapipe_landmarks is None:
        return None
    
    # refine_landmarks=True일 때 478개가 반환되므로, 처음 468개만 사용
    if mediapipe_landmarks.shape[0] == 478:
        mediapipe_landmarks = mediapipe_landmarks[:468]  # 처음 468개만 사용
    elif mediapipe_landmarks.shape[0] != 468:
        print(f"Warning: Expected 468 or 478 landmarks, got {mediapipe_landmarks.shape[0]}")
        return None
    
    # 인덱스로 선택 (2D만 사용)
    if mediapipe_landmarks.shape[1] >= 2:
        landmarks_68 = mediapipe_landmarks[MEDIAPIPE_TO_68_INDICES, :2]  # [68, 2]
    else:
        landmarks_68 = mediapipe_landmarks[MEDIAPIPE_TO_68_INDICES]  # [68]
        landmarks_68 = landmarks_68.reshape(-1, 1)  # [68, 1] -> [68, 2]로 확장 필요
    
    return landmarks_68


def detect_landmarks_from_image(image, face_mesh=None):
    """
    렌더링된 이미지에서 MediaPipe를 사용하여 랜드마크 검출
    
    Args:
        image: numpy array, shape [H, W, 3], RGB, 0-255 범위
        face_mesh: MediaPipe FaceMesh 객체 (None이면 새로 생성)
    
    Returns:
        landmarks: numpy array, shape [468, 2] 또는 None (얼굴 미검출 시)
    """
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    if face_mesh is None:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
    
    # MediaPipe는 RGB 이미지를 요구
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 이미 RGB인 경우
        rgb_image = image
    else:
        # BGR to RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # uint8로 변환 (0-255 범위)
    if rgb_image.dtype != np.uint8:
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_image = rgb_image.astype(np.uint8)
    
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        return None
    
    # 첫 번째 얼굴의 랜드마크 추출
    face_landmarks = results.multi_face_landmarks[0]
    
    # MediaPipe landmark를 numpy array로 변환 [468 or 478, 3] (x, y, z)
    # refine_landmarks=True일 때 478개가 반환되므로, 처음 468개만 사용
    num_landmarks = len(face_landmarks.landmark)
    landmarks = np.zeros((num_landmarks, 3))
    for i, landmark in enumerate(face_landmarks.landmark):
        landmarks[i, 0] = landmark.x * rgb_image.shape[1]  # x 좌표 (픽셀 단위)
        landmarks[i, 1] = landmark.y * rgb_image.shape[0]  # y 좌표 (픽셀 단위)
        landmarks[i, 2] = landmark.z  # z 좌표 (깊이)
    
    # refine_landmarks=True일 때 478개가 반환되므로, 처음 468개만 사용
    if num_landmarks == 478:
        landmarks = landmarks[:468]
    
    # 2D landmark만 반환 [468, 2]
    return landmarks[:, :2]


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    # 모델 경로에서 마지막 디렉토리 이름 추출하여 출력 경로 자동 생성
    # savefolder가 기본값이거나 *를 포함하는 경우에만 자동 생성
    # 명시적으로 지정된 경우는 그대로 사용
    if args.savefolder == '/media/cine/First/HWPJ2/ProjectResult/AFWE_VA/test1_detail/*/' or '*' in args.savefolder:
        model_path = args.pretrained_modelpath_ViT
        # 모델 경로에서 디렉토리 이름 추출
        # 예: /media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune/model.tar -> DetailNew_FineTune
        model_dir = os.path.basename(os.path.dirname(model_path))
        # 출력 경로 생성: AFWE_VA 아래에 모델 디렉토리 이름 사용
        args.savefolder = f'/media/cine/First/HWPJ2/ProjectResult/AFWE_VA/{model_dir}/*/'
        print(f"Auto-generated savefolder from model path: {args.savefolder}")
    else:
        print(f"Using specified savefolder: {args.savefolder}")

    # MediaPipe FaceMesh 초기화
    if MEDIAPIPE_AVAILABLE:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
    else:
        face_mesh = None
        print("Warning: MediaPipe not available. Detail landmark detection will be skipped.")

    allImageFlord = sorted(glob.glob(args.inputpath))
    for imageDir in allImageFlord:
        # for k in range(len(actors)):
            # savefolder =
            partpath = imageDir.split("croppedImages/")[-1]
            savefolder = args.savefolder.replace("*",partpath[:-1])
            vidoname = args.vidoname.replace("*",partpath[:-1].replace("/","_"))
            if os.path.exists(savefolder):
                print("exists...")
                continue
            os.makedirs(savefolder, exist_ok=True)
            # os.makedirs(imageDir.replace("croppedImages","tform"), exist_ok=True)

            # load test images
            # testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
            testdata = datasets.TestData(imageDir, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(os.path.join(savefolder, vidoname + ".mp4"), fourcc, 30, (448 * 6, 448), True)
            #
            # writer = pd.ExcelWriter(
            #         os.path.join(savefolder, 'parameters.xlsx'))
            # # for i in range(len(testdata)):
            # writeContent = []

            os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
            # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, '2d_landmark_68'), exist_ok=True)

            for i in tqdm(range(1, len(testdata) - 1)):
                data_1 = testdata[i - 1]
                data = testdata[i]
                data_3 = testdata[i + 1]
                name = data['imagename']


                images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                                   0).to(device)

                # data = testdata[i]
                # name = data['imagename']

                # images = testdata[i]['image'].to(device)[None,...]

                with torch.no_grad():
                    codedict_old, codedict = deca.encode(images)
                    # codedict['pose'][0][:3] = 0.
                    # codedict['cam'] = codedict_deca['cam']
                    # use_detail=True로 변경하여 detail rendering 활성화
                    opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True)  # tensor

                    if args.render_orig:
                        tform = testdata[i]['tform'][None, ...]
                        tform = torch.inverse(tform).transpose(1, 2).to(device)
                        original_image = testdata[i]['original_image'][None, ...].to(device)
                        _, orig_visdict = deca.decode(codedict, codedict_old, render_orig=True,
                                                      original_image=original_image, tform=tform)
                        orig_visdict['inputs'] = original_image

                # if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
                #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'landmark_7'), exist_ok=True)
                # -- save results
                # tform
                if args.saveKpt:
                    # 기존 coarse mesh에서 계산한 랜드마크 저장
                    np.save(os.path.join(savefolder, "2d_landmark_68", name + '.npy'),
                               opdict['landmarks2d'][0].cpu().numpy())
                    np.save(os.path.join(savefolder, "2d_landmark_68", name + '_DECA.npy'),
                               opdict['landmarks2d_old'][0].cpu().numpy())
                    
                    # Detail 이미지에서 랜드마크 검출
                    if MEDIAPIPE_AVAILABLE and 'shape_detail_images' in opdict:
                        # Detail rendering된 이미지 가져오기
                        detail_image = opdict['shape_detail_images'][0].cpu().numpy()  # [3, H, W]
                        
                        # [H, W, 3] 형태로 변환하고 RGB로 변환
                        detail_image = detail_image.transpose(1, 2, 0)  # [H, W, 3]
                        
                        # 텐서를 numpy로 변환하고 0-1 범위를 0-255로 변환
                        if detail_image.max() <= 1.0:
                            detail_image_uint8 = (detail_image * 255).astype(np.uint8)
                        else:
                            detail_image_uint8 = detail_image.astype(np.uint8)
                        
                        # MediaPipe로 랜드마크 검출 (468개)
                        landmarks2d_detail_mp = detect_landmarks_from_image(detail_image_uint8, face_mesh)
                        
                        if landmarks2d_detail_mp is not None:
                            # MediaPipe 468개 landmark를 68개로 매핑
                            landmarks2d_detail_68 = map_mediapipe_to_68(landmarks2d_detail_mp)
                            
                            if landmarks2d_detail_68 is not None:
                                # 68개로 매핑된 랜드마크 저장 (calculateError_detail.py에서 사용)
                                np.save(os.path.join(savefolder, "2d_landmark_68", name + '_detail.npy'),
                                       landmarks2d_detail_68)
                                # 원본 468개도 저장 (참고용)
                                np.save(os.path.join(savefolder, "2d_landmark_68", name + '_detail_mediapipe.npy'),
                                       landmarks2d_detail_mp)
                                print(f"Detected and mapped {landmarks2d_detail_68.shape[0]} landmarks from detail image for {name}")
                            else:
                                print(f"Warning: Failed to map MediaPipe landmarks for {name}")
                        else:
                            print(f"Warning: No landmarks detected from detail image for {name}")
                    elif not MEDIAPIPE_AVAILABLE:
                        print(f"Warning: MediaPipe not available, skipping detail landmark detection for {name}")
                    elif 'shape_detail_images' not in opdict:
                        print(f"Warning: shape_detail_images not found in opdict for {name}")
                    
                    # np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'),
                    #            opdict['landmarks3d'][0].cpu().numpy())
                if args.saveObj:
                    os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
                    os.makedirs(os.path.join(savefolder, 'landmark_7'), exist_ok=True)

                    deca.save_obj(os.path.join(savefolder, 'obj', name + '.obj'), opdict)
                    landmark_51 = opdict['landmarks3d_world'][:, 17:]
                    landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
                    landmark_7 = landmark_7.cpu().numpy()
                    np.save(os.path.join(savefolder, 'landmark_7', name + '.npy'), landmark_7[0])

                # cv2.imwrite(os.path.join(savefolder, 'result',name + '.jpg'), deca.visualize(visdict))
                vis_image = deca.visualize(visdict, size=448)
                # orig_vis_image = deca.visualize(orig_visdict, size=448)

                cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)
                # cv2.imwrite(os.path.join(savefolder, 'result_original', name + '.jpg'), orig_vis_image)
                # cropPath = os.path.join(inputpath.replace("originalImages", "croppedImages"), name + '.jpg')
                # if not os.path.exists(cropPath):
                #     cv2.imwrite(cropPath,
                #             deca.visualize({"image": data['image'][None, ...]}, size=224))
                out.write(vis_image)
            print(f'-- please check the results in {savefolder}')
            out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation with Detail Landmark Detection')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="*",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Downloads/AFEW-VA/croppedImages/*/*/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain2/Actor_id/exp_texture/',
                        default='/media/cine/First/HWPJ2/ProjectResult/AFWE_VA/test1_detail/*/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_2/pretrain4/*/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1/models/00820199.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X/model.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X1/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain2/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain4/model.tar',
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain5X/17epoch.tar',

                        type=str,
                        help='model.tar path')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step')
    parser.add_argument('--detector', default='retinaface', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard')
    parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--extractTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode')
    parser.add_argument('--saveVis', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())
