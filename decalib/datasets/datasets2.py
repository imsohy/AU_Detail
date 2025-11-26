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
import copy
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize, rescale
import glob


class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.05, face_detector='retinaface',
                 ifCenter='', ifSize=''):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        self.mediapipe_idx = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336,  46,  52,  53,  55,  63,  65,  66,  70,
105, 107, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466,
  7,  33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 168,   6,
197, 195,   5, 4, 129,  98,  97,   2, 326, 327, 358,   0,  13,  14,  17,  37,  39,  40,
 61,  78,  80, 81,  82,  84,  87,  88,  91,  95, 146, 178, 181, 185, 191, 267, 269, 270,
291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

        self.imagepath_list = glob.glob(testpath + '/*.jpg') + glob.glob(testpath + '/*.png') + glob.glob(
            testpath + '/*.bmp')
        self.type = 'image'

        try:
            self.imagepath_list = sorted(self.imagepath_list,
                                         key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0].split('frame')[-1]))
        except:
            self.imagepath_list = sorted(self.imagepath_list)

        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size


    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape

        # lmkpath = imagepath.replace('images','kpts').replace('.png','.npy').replace('.jpg','.npy')
        # lmk_densepath = lmkpath.replace('kpts','kpts_dense')

        src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)


        image = image / 255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

        # lmk = np.load(lmkpath, allow_pickle=True)
        # lmk_dense = np.load(lmk_densepath, allow_pickle=True)


        dst_image = dst_image.transpose(2, 0, 1)
        return {
                'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(),
                'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                'imagepath': imagepath,
                # 'lmks':torch.tensor(lmk).float(),
                # 'lmks_dense':torch.tensor(lmk_dense[self.mediapipe_idx, :]).float(),
                }