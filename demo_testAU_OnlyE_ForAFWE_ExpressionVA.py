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
from decalib.models.expression_loss import ExpressionLossNet

# import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.gatfarec_Video_OnlyExpress import DECA
from decalib.datasets import datasets as datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def main(args):
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)

    allImageFlord = sorted(glob.glob(args.inputpath))
    for imageDir in allImageFlord:
            partpath = imageDir.split("croppedImages/")[-1]
            savefolder = args.savefolder.replace("*",partpath[:-1])
            vidoname = args.vidoname.replace("*",partpath[:-1].replace("/","_"))
            # if os.path.exists(savefolder):
            #     print("exists...")
            #     continue
            os.makedirs(savefolder, exist_ok=True)
            testdata = datasets.TestData(imageDir, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )
            # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            # out = cv2.VideoWriter(os.path.join(savefolder, vidoname + ".mp4"), fourcc, 30, (448 * 6, 448), True)
            #
            expression_net = ExpressionLossNet().to(device)
            emotion_checkpoint = torch.load('/home/cine/Documents/HJCode/AU_sequence/data/dataloader_idx_0=1.27607644.ckpt')['state_dict']
            emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
            emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
            expression_net.load_state_dict(emotion_checkpoint, strict=False)
            expression_net.eval()
            os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
            # os.makedirs(os.path.join(savefolder, '2d_landmark_68'), exist_ok=True)
            # tempName =
            f = open("/home/cine/Desktop/AFEW_VAResult/"+partpath.split('/')[-2]+".txt", 'a')
            fe = open("/home/cine/Desktop/AFEW_VAResult/"+partpath.split('/')[-2]+"_e.txt", 'a')
            for i in tqdm(range(1, len(testdata) - 1)):
                data_1 = testdata[i - 1]
                data = testdata[i]
                data_3 = testdata[i + 1]
                name = data['imagename']
                if os.path.exists(os.path.join(savefolder, 'result', name + '.jpg')):
                    print("exists...")
                    continue

                images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                                   0).to(device)

                with torch.no_grad():
                    codedict_old, codedict = deca.encode(images)
                    # codedict['pose'][0][:3] = 0.
                    # codedict['cam'] = codedict_deca['cam']
                    opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor

                _,emotion_pred = expression_net.forward2(opdict['rendered_images'])
                _, emotion_pred_emoca = expression_net.forward2(opdict['rendered_images_emoca'])
                # print(emotion_pred.detach().cpu())
                # print(emotion_pred_emoca.detach().cpu())
                f.write(name+" "+str(emotion_pred.detach().cpu().tolist()[0][-2])+" "+str(emotion_pred.detach().cpu().tolist()[0][-1])+"\n")
                fe.write(name+" "+str(emotion_pred_emoca.detach().cpu().tolist()[0][-2])+" "+str(emotion_pred_emoca.detach().cpu().tolist()[0][-1])+"\n")
                # cv2.imwrite(os.path.join(savefolder, 'result',name + '.jpg'), deca.visualize(visdict))
                # vis_image = deca.visualize(visdict, size=448)
                # orig_vis_image = deca.visualize(orig_visdict, size=448)

                # cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)
                # cv2.imwrite(os.path.join(savefolder, 'result_original', name + '.jpg'), orig_vis_image)
                # cropPath = os.path.join(inputpath.replace("originalImages", "croppedImages"), name + '.jpg')
                # if not os.path.exists(cropPath):
                #     cv2.imwrite(cropPath,
                #             deca.visualize({"image": data['image'][None, ...]}, size=224))
                # out.write(vis_image)
            print(f'-- please check the results in {savefolder}')
            # out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="*",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Downloads/AFEW-VA/croppedImages/*/*/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/*/',
                        default='/home/cine/Documents/DDD/TestReult_New0420/AULoss/sequence_pretrain6/AFWE_VA/*/',

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
