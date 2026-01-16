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
from decalib.models.OpenGraphAU.model.MEFL_o import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.conf_DISFA import get_config,set_env
from datetime import datetime


from decalib.models.OpenGraphAU.model.MEFL import MEFARG as MEFARG_27
# from decalib.models.OpenGraphAU.utils import load_state_dict as
from decalib.models.OpenGraphAU.conf import get_config as get_config_27
# from utils_MG import statistics, update_statistics_list,



    # return sAU, dAU

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


    au_labels_27 = ["au1", "au2", "au4", "au5", "au6","au7", "au9","au10","au11",
                 "au12", "au13","au14","au15","au16", "au17", "au18","au19",
                 "au20","au22","au23","au24", "au25", "au26","au27","au32","au38","au39"]
    auconf_27 = get_config_27()

    auconf_27.evaluate = True
    auconf_27.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]

    set_env(auconf_27)
    AU_net_27 = MEFARG_27(num_main_classes=auconf_27.num_main_classes, num_sub_classes=auconf_27.num_sub_classes,
                         backbone=auconf_27.arc).to(device)
    AU_net_27 = load_state_dict(AU_net_27, auconf_27.resume).to(device)
    AU_net_27.eval()
    savefolder = args.savefolder

    os.makedirs(savefolder, exist_ok=True)

    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                 scale=1.25, )

    os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)

    for i in tqdm(range(len(testdata))):
        data_1 = testdata[i - 1]
        data = testdata[i]
        print((i+1)%len(testdata))
        data_3 = testdata[(i+1)%len(testdata)]

        name = data['imagename']

        images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                           0).to(device)

        fileName2 = os.path.join(savefolder, "au27", name.split("LeftVideo")[-1].split("_")[0])

        with torch.no_grad():
            print(datetime.now)
            codedict_old, codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor

            print(datetime.now)
            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1, 2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, codedict_old, render_orig=True,
                                              original_image=original_image, tform=tform)
                orig_visdict['inputs'] = original_image
        image_au = AU_net_27(images[1:2])
        rend_au = AU_net_27(opdict['rendered_images'])
        rend_au_deca = AU_net_27(opdict['rendered_images_emoca'])
        print(rend_au[1].float()[0])
        # opdict['au_img'] = (image_au[1] >= 0.5).float()[0]
        # opdict['au_rend'] = (rend_au[1] >= 0.5).float()[0]
        # opdict['au_rend_deca'] = (rend_au_deca[1] >= 0.5).float()[0]
        # resultI, resultR = vis_au(opdict['au_img'] , opdict['au_rend'] )
        # for kt, xt in enumerate(au_labels_27):
        #     # print(opdict['au_img'][kt], opdict['au_rend'][kt])
        #     with open(fileName2 + "_" + xt + ".txt", "a") as f:
        #         f.write(str(opdict['au_img'].cpu().numpy().tolist()[kt]) + "\n")
        #     with open(fileName2 + "_" + xt + "R.txt", "a") as f:
        #         f.write(str(opdict['au_rend'].cpu().numpy().tolist()[kt]) + "\n")
        #     with open(fileName2 + "_" + xt + "R_emoca.txt", "a") as f:
        #         f.write(str(opdict['au_rend_deca'].cpu().numpy().tolist()[kt]) + "\n")

        vis_image = deca.visualize(visdict, size=448)

        cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)

    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] ="0,3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry_same/",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry_sameF/",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry2_same/",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry3_same/",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry3_same/",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry/",
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/differentID/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        default='/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain6/ForCheck/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_CropBEmoca/pretrain5X_25/*/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_2/pretrain4/*/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain6/model.tar',
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
