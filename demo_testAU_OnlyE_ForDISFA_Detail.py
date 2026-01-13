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

# ============================================================================
# 수정 사항: Detail 모델을 사용한 AU 감지 코드
# 기반 파일: demo_testAU_OnlyE_ForDISFA_AU1_27.py
# 주요 변경: use_detail=True로 설정하여 Detail 모델 활성화
# ============================================================================

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
from decalib.gatfarec_Video_OnlyExpress import DECA
from decalib.datasets import datasets as datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.models.OpenGraphAU.model.MEFL import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.conf import get_config,set_logger,set_outdir,set_env

# def vis_au(srcAU, dstAU):
#
#     # AU 라벨 리스트
#     au_labels = ["AU1", "AU2", "AU4", "AU5", "AU7", "AU9",
#                  "AU10", "AU12", "AU15", "AU20", "AU23", "AU26",
#                  "AU1", "AU2", "AU4", "AU5", "AU7", "AU9",
#                  "AU10", "AU12", "AU15", "AU20", "AU23", "AU26"]



    # return sAU, dAU

def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    # compare1 = [0, 1, 2, 3, 5, 6, 7, 9, 12, 17, 19, 22]
    # compare2 = [0, 1, 3, 4, 5, 8, 11, 14, 16, 19, 24, 25]

    # AU 라벨 정의 (27개 AU 중 일부 제외)
    au_labels = ["au1", "au2", "au4", "au5", "au6","au7", "au9","au10","au11",
                 "au12", "au13","au14","au15","au16", "au17", "au18","au19",
                 "au20","au22","au23","au24", "au25", "au26","au27"]
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    
    # DECA 모델 초기화
    deca = DECA(config=deca_cfg, device=device)
    
    # MEFARG (AU 감지 모델) 초기화
    auconf = get_config()
    auconf.evaluate = True
    auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    set_env(auconf)
    AU_net = MEFARG(num_main_classes=auconf.num_main_classes, num_sub_classes=auconf.num_sub_classes,
                         backbone=auconf.arc).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()
    
    # 비디오 경로 목록 가져오기
    allVideos = sorted(glob.glob(args.inputpath), reverse=True)
    for videopath in allVideos:
        # for k in range(len(actors)):
            # savefolder =
            inputpath = videopath
            name = videopath.split("/croppedImages2")[0].split("/")[-1]
            savefolder = args.savefolder.replace("*",name)
            fileName1 = os.path.join(savefolder, "au27",name.split("LeftVideo")[-1].split("_")[0])
            # fileName2 = os.path.join(savefolder, "au2",name.split("LeftVideo")[-1].split("_")[0])
            # vidoname = args.vidoname.replace("*",name)
            # if os.path.exists(savefolder):
            #     print("exists...")
            #     continue
            os.makedirs(savefolder, exist_ok=True)
            if os.path.exists(os.path.join(savefolder, "au27")):
                continue

            # 테스트 이미지 로드
            # testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
            testdata = datasets.TestData(inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )
            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # out = cv2.VideoWriter(os.path.join(savefolder, vidoname + ".mp4"), fourcc, 30, (448 * 6, 448), True)
            #
            # writer = pd.ExcelWriter(
            #         os.path.join(savefolder, 'parameters.xlsx'))
            # # for i in range(len(testdata)):
            # writeContent = []

            # 결과 저장 디렉토리 생성
            os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, 'au27'), exist_ok=True)
            # os.makedirs(os.path.join(savefolder, 'au2'), exist_ok=True)
            os.makedirs(os.path.splitext(inputpath)[0].replace("originalImages", "croppedImages"),exist_ok=True )

            # 각 프레임에 대해 처리 (3프레임 시퀀스 사용)
            for i in tqdm(range(1, len(testdata) - 1)):
                data_1 = testdata[i - 1]
                data = testdata[i]
                data_3 = testdata[i + 1]
                name = data['imagename']

                # 3프레임 이미지를 하나의 텐서로 결합
                images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                                   0).to(device)

                # data = testdata[i]
                # name = data['imagename']

                # images = testdata[i]['image'].to(device)[None,...]

                with torch.no_grad():
                    # DECA 인코딩: 얼굴 파라미터 추출
                    codedict_old, codedict = deca.encode(images)
                    # codedict['pose'][0][:3] = 0.
                    # codedict['cam'] = codedict_deca['cam']
                    
                    # ========================================================================
                    # 수정 사항 1: use_detail=True로 변경하여 Detail 모델 활성화
                    # 기존: use_detail=False (Coarse 모델만 사용)
                    # 변경: use_detail=True (Detail 모델 포함, 주름 및 미세한 표정 변화 포함)
                    # 참고: return_vis=True (기본값)로 설정하여 visdict에서 Detail 결과를 받아옴
                    # ========================================================================
                    opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True, return_vis=True)  # tensor
                    # 중요: opdict['rendered_images']는 항상 Coarse 결과입니다
                    # Detail 결과는 visdict['render_images_with_detail']에 저장됩니다
                    # (주름, 미세한 표정 변화 등 Detail 정보 포함)

                    if args.render_orig:
                        tform = testdata[i]['tform'][None, ...]
                        tform = torch.inverse(tform).transpose(1, 2).to(device)
                        original_image = testdata[i]['original_image'][None, ...].to(device)
                        _, orig_visdict = deca.decode(codedict, codedict_old, render_orig=True,
                                                      original_image=original_image, tform=tform)
                        orig_visdict['inputs'] = original_image

                # AU 감지 수행
                # 원본 이미지의 AU 감지
                image_au = AU_net(images[1:2])
                
                # ========================================================================
                # 수정 사항 3: Detail 재구성 이미지 사용
                # 기존: opdict['rendered_images'] (Coarse 결과만 포함)
                # 변경: visdict['render_images_with_detail'] (Coarse + Detail 결과 포함)
                # 근거: decode 함수에서 Detail 결과는 visdict['render_images_with_detail']에만 저장됨
                #      (gatfarec_Video_DetailNewBranch_v3.py 585번째 줄 참조)
                # ========================================================================
                rend_au = AU_net(visdict['render_images_with_detail'])
                
                # ========================================================================
                # 수정 사항 2: rendered_images_deca → rendered_images_old로 변경
                # 기존: opdict['rendered_images_deca'] (존재하지 않는 키)
                # 변경: opdict['rendered_images_old'] (DECA baseline의 Coarse 재구성 이미지)
                # 참고: DECA baseline은 Coarse만 사용하므로 Detail 정보는 없습니다
                # ========================================================================
                rend_au_deca = AU_net(opdict['rendered_images_old'])

                # AU 예측 결과를 이진화 (threshold=0.5)
                opdict['au_img'] = (image_au[1] >= 0.5).float()[0]  # 원본 이미지 AU
                opdict['au_rend'] = (rend_au[1] >= 0.5).float()[0]  # 우리 모델 Detail AU
                opdict['au_rend_deca'] = (rend_au_deca[1] >= 0.5).float()[0]  # DECA baseline Coarse AU
                # resultI, resultR = vis_au(opdict['au_img'] , opdict['au_rend'] )
                
                # 각 AU에 대한 결과를 텍스트 파일로 저장
                for kt, xt in enumerate(au_labels):
                    # print(opdict['au_img'][kt], opdict['au_rend'][kt])
                    # 원본 이미지 AU 결과 저장
                    with open(fileName1+"_"+xt+".txt", "a") as f:
                        f.write(str(opdict['au_img'].cpu().numpy().tolist()[kt])+"\n")
                    # 우리 모델 Detail AU 결과 저장
                    with open(fileName1+"_"+xt+"R.txt", "a") as f:
                        f.write(str(opdict['au_rend'].cpu().numpy().tolist()[kt])+"\n")
                    # DECA baseline Coarse AU 결과 저장
                    with open(fileName1+"_"+xt+"R_deca.txt", "a") as f:
                        f.write(str(opdict['au_rend_deca'].cpu().numpy().tolist()[kt])+"\n")

                # if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
                #     os.makedirs(os.path.join(savefolder, name), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'obj'), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
                # os.makedirs(os.path.join(savefolder, 'landmark_7'), exist_ok=True)
                # -- save results
                if args.saveDepth:
                    depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
                    visdict['depth_images'] = depth_image
                    cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
                if args.saveKpt:
                    np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'),
                               opdict['landmarks2d'][0].cpu().numpy())
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

                # vis_image = deca.visualize(visdict, size=448)

                # cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)
            # print(f'-- please check the results in {savefolder}')
            # out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation (Detail 모델 사용)')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="*",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video/*/croppedImages2/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain2/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA/AULoss1_ELT/*/',
                        default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_2/pretrain5X_25_Detail/*/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_2/pretrain4/*/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1/models/00820199.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X/model.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X1/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain2/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain4/model.tar',
                        default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain5X/models/00410099.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain1/models/00410099.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain5X/17epoch.tar',

                        type=str,
                        help='model.tar path')
    parser.add_argument('--device', default='cuda:1', type=str,
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
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
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

