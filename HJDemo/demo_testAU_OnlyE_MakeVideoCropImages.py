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
from decalib.gatfarec_Video_OnlyExpress import DECA
from decalib.datasets import datasets as datasets
from decalib.utils import util
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

    allImageFlord = sorted(glob.glob(args.inputpath))
    for imageDir in allImageFlord:
            # inputpath =os.path.join(imageDir,'*.*')
            print(imageDir)
            # if os.path.exists(os.path.splitext(videopath)[0].replace("originalImages", "croppedImages3")):
            savepath = imageDir.replace("images", "croppedImages")
            # if os.path.exists(savepath):
            #     print("exists")
            #     continue
            # os.makedirs(os.path.splitext(videopath)[0].replace("originalImages", "croppedImages3"),exist_ok=True )
            os.makedirs(savepath,exist_ok=True )
            os.makedirs(imageDir.replace("images", "tform"),exist_ok=True )


            testdata = datasets.TestData(imageDir, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )
            for i in tqdm(range(len(testdata))):
                data = testdata[i]
                name = data['imagename']
                # cropPath = os.path.join(videopath.replace("originalImages", "croppedImages3"), name + '.jpg')
                cropPath = os.path.join(imageDir.replace("images", "croppedImages"), name + '.jpg')
                if not os.path.exists(cropPath):
                    cv2.imwrite(cropPath,
                            deca.visualize({"image": data['image'][None, ...]}, size=224))
                tformpath = cropPath.replace("croppedImages", "tform").replace(".jpg", ".npy").replace(".png",
                                                                                                                ".npy")
                tform = data['tform']
                np.save(tformpath, tform)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="*",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    # parser.add_argument('-i', '--inputpath', default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video/*/originalImages/",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Downloads/AFEW-VA/images/*/*",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain2/Actor_id/exp_texture/',
                        default='',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1/models/00820199.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X/model.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X1/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain2/model.tar',
                        default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain4/model.tar',
                        type=str,
                        help='model.tar path')
    parser.add_argument('--device', default='cuda:1', type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
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
