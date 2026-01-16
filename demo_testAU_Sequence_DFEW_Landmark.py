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
import shutil
import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

# import pandas as pd
from glob import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.gatfarec_Video_OnlyExpress_AFEW import DECA
from decalib.datasets import datasets_GetFAN as datasets
from decalib.utils.config import cfg as deca_cfg
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
    NOFACE = 0

    allVideos = sorted(glob(args.inputpath))
    for videopath in allVideos:
            inputpath = glob(os.path.join(videopath, '*.jpg'))

            video_name = videopath.split('/')[-2]
            savefolder = args.savefolder.replace("name", video_name)

            os.makedirs(savefolder, exist_ok=True)
            if os.path.exists(os.path.join(savefolder, 'forExp_EMOCA')):
                print(savefolder)
                # del
                continue
            if os.path.exists(os.path.join(savefolder, 'au27')):
                shutil.rmtree(os.path.join(savefolder, 'au27'))
            testdata = datasets.TestData(inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )
            os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
            os.makedirs(os.path.join(savefolder,'2d_landmark_68'), exist_ok=True)

            for i in tqdm(range(len(testdata) )):
                data_1 = testdata[i - 1]
                data = testdata[i]
                if i+1==len(testdata):
                    data_3 = testdata[0]
                else:
                    data_3 = testdata[i + 1]
                if data_1==0 or data ==0 or data_3==0:
                    NOFACE+=1
                    continue

                name = data['imagename']

                images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                                   0).to(device)

                lmk = data['landmark'].to(device)
                lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])

                # data = testdata[i]
                # name = data['imagename']

                # images = testdata[i]['image'].to(device)[None,...]

                with torch.no_grad():
                    codedict_old, codedict = deca.encode(images)
                    # codedict['pose'][0][:3] = 0.
                    # codedict['cam'] = codedict_deca['cam']
                    # opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor
                    opdict, visdict = deca.decode(lmk,codedict, codedict_old, use_detail=False)  # tensor


                np.save(os.path.join(savefolder, "2d_landmark_68", name + '_FAN.npy'),
                           lmk.cpu().numpy())
                np.save(os.path.join(savefolder, "2d_landmark_68", name + '_Ours.npy'),
                           opdict['landmarks2d'][0].cpu().numpy())
                np.save(os.path.join(savefolder, "2d_landmark_68", name + '_EMOCA.npy'),
                           opdict['landmarks2d_old'][0].cpu().numpy())
                vis_image = deca.visualize(visdict, size=448)
                cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)

            print(f'-- please check the results in {savefolder}')
            # out.release()

    print(NOFACE)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    # parser.add_argument('-vn', '--vidoname', default="Actor_idexp",
    #                     type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/croppedImages1_1_2/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Downloads/DFEW/DFEW-part2/Clip/clip_224x224/clip_224/*/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain5_15/Actor_id/exp_texture/',
                        default='/media/cine/First/ForPaperResult/LandmarkVS/LandmarkVS/sequence_pretrain6_2/DFEW_/name/',

                        # default='/home/cine/Documents/ForPaperResult/TestReult/AULoss_CropE/pretrain/Actor_id/exp_texture/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        # default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain5/model.tar',
                        # default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain5/models/00246059.tar',
                        default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain6/model.tar',

                        # default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULossF/pretrain1/model.tar',
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
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
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
