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
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
# import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.gatfarec_Video_OnlyExpress_image import DECA
from decalib.datasets import datasets as datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    # savefolder = args.savefolder
    device = args.device
    # os.makedirs(savefolder, exist_ok=True)

    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                 scale=1.15, )
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
    savefolder = args.inputpath.replace("actors","actors/croppedImage")
    os.makedirs(os.path.join(savefolder, 'croppedImage'), exist_ok=True)
    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    for i in tqdm(range(0, len(testdata))):
        # data_1 = testdata[i - 1]
        data = testdata[i]
        # data_3 = testdata[i + 1]
        name = data['imagename']

        images =  data['image'][None, ...].to(device)
        vis_image = deca.visualize({'imaegs':images}, size=224)
        cv2.imwrite(os.path.join(savefolder,  name + '.jpg'), vis_image)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
    # # 08, 30, 37, 43, 64, 75
    # name = 'AFLW2000_3d'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    name = 'Actor_03angry'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # name = 'ForModelImage'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # name = '002645310-'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # sentence = 'SEN_approach_your_interview_with_statuesque_composure'
    # name+=sentence
    # parser.add_argument('-i', '--inputpath', default='/media/cine/First/TestDataset/images/FaMoS_subject_064/', type=str,
    parser.add_argument('-vn', '--vidoname', default=name, type=str, )  # # 05happy 14  16calm  18disgust 18sad
    parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/actors/' + name, type=str,
    # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/actors/' + name, type=str,
    # parser.add_argument('-i', '--inputpath', default='/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/71/*.png', type=str,
    # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/AFLW2000-3D/AFLW2000/*.jpg', type=str,
    # parser.add_argument('-i', '--inputpath', default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI_7_17/multiface/mini_datasetB/m--20190828--1318--002645310--GHS/images2/'+sentence+'/*', type=str,
                        # 05happy 14  16calm  18disgust 18sad
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestReult/OnlyE_image/pretrain1_1/' + name + '_texture2/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE_image/pretrain1/model.tar', type=str,
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
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())