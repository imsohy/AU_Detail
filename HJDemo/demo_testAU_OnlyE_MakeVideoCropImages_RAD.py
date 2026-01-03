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
from decalib.datasets import datasets as datasets
from decalib.utils.config import cfg as deca_cfg
import torch
import torchvision
from torch.nn import functional as F

def visualize(visdict, size=224, dim=2):
    '''
    image range should be [0,1]
    dim: 2 for horizontal. 1 for vertical
    '''
    assert dim == 1 or dim == 2
    grids = {}
    for key in visdict:
        _, _, h, w = visdict[key].shape
        if dim == 2:
            new_h = size;
            new_w = int(w * size / h)
        elif dim == 1:
            new_h = int(h * size / w);
            new_w = size
        grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [new_h, new_w]).detach().cpu())

    grid = torch.cat(list(grids.values()), dim)
    grid_image = (grid.numpy().transpose(1, 2, 0).copy() * 255)[:, :, [2, 1, 0]]
    grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
    return grid_image
def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    actors = sorted(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12",
              "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", ],reverse=True)
    # expressions = ["01", "02", "03", "04", "05", "06", "07", "08"]
    expressionNames = ["angry", "happy", "sad", "fearful", "disgust", "surprised","neural", "calm"]
    for j in range(len(expressionNames)):
        for k in range(len(actors)):
            inputpath = args.inputpath.replace("id", actors[k]).replace("exp", expressionNames[j])
            testdata = datasets.TestData(os.path.join(inputpath,"*.*"), iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,face_detector = "retinaface",
                                         scale=1.1, )
            # testdata = datasets.TestData(os.path.join(inputpath,"*.*"), iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,face_detector = "fan",
            #                              scale=1.1, )
            if os.path.exists(os.path.splitext(inputpath)[0].replace("originalImages", "croppedImages1_1_2")):
                print("exists")
                continue
            os.makedirs(os.path.splitext(inputpath)[0].replace("originalImages", "croppedImages1_1_2"),exist_ok=True )

            for i in tqdm(range(1, len(testdata)-1)):
                data = testdata[i]
                name = data['imagename']

                cropPath = os.path.join(inputpath.replace("originalImages", "croppedImages1_1_2"), name + '.jpg')
                if not os.path.exists(cropPath):
                    cv2.imwrite(cropPath,
                            visualize({"image": data['image'][None, ...]}, size=224))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="Actor_idexp",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/croppedImages/Actor_id/exp",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain4/Actor_id/exp_texture/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
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
    parser.add_argument('--detector', default='fan', type=str, # fan, retinaface
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
