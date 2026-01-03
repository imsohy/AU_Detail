"""
2025-12-20 23:00작성.
이전 버전: demo_testAU_OnlyE_ForAFWE.py
번경 이유:
어떻게 원본 demo파일을 구동시키려고 해서, 하드코딩된거 좀 고쳤던 파일이다.
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
                    opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor

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
                    np.save(os.path.join(savefolder, "2d_landmark_68", name + '.npy'),
                               opdict['landmarks2d'][0].cpu().numpy())
                    np.save(os.path.join(savefolder, "2d_landmark_68", name + '_DECA.npy'),
                               opdict['landmarks2d_old'][0].cpu().numpy())
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
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
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
                        default='/media/cine/First/HWPJ2/ProjectResult/AFWE_VA/test1/*/',
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
