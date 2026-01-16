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
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.gatfarec_Video_OnlyExpress import DECA
from decalib.datasets import datasets as datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
from decalib.models.OpenGraphAU.model.MEFL_o import MEFARG
from decalib.models.OpenGraphAU.utils import load_state_dict
from decalib.models.OpenGraphAU.conf_DISFA import get_config, set_env
from datetime import datetime

from decalib.models.OpenGraphAU.model.MEFL import MEFARG as MEFARG_27
# from decalib.models.OpenGraphAU.utils import load_state_dict as
from decalib.models.OpenGraphAU.conf import get_config as get_config_27
# from utils_MG import statistics, update_statistics_list,

from decalib.models.expression_loss import ExpressionLossNet


def add_labels(rects):
    for rect in rects:
        # height = rect.get_height()
        width = rect.get_width()
        # x = rect.get_x()
        # y = rect.get_y()
        # print(height, width,x,y)
        plt.text(width, rect.get_y() + rect.get_width() / 2, round(width,3), ha='left', va='center')
        # plt.text(rect.get_y() + height() / 2, width, width, ha='center', va='bottom')
        rect.set_edgecolor('white')
# return sAU, dAU
import matplotlib.pyplot as plt
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

    expression_net = ExpressionLossNet().to(device)
    emotion_checkpoint = torch.load('/home/cine/Documents/HJCode/AU_sequence/data/dataloader_idx_0=1.27607644.ckpt')[
        'state_dict']
    emotion_checkpoint['linear.0.weight'] = emotion_checkpoint['linear.weight']
    emotion_checkpoint['linear.0.bias'] = emotion_checkpoint['linear.bias']
    expression_net.load_state_dict(emotion_checkpoint, strict=False)
    expression_net.eval()
    au_labels_27 = ["au1", "au2", "au4", "au5", "au6", "au7", "au9", "au10", "au11",
                    "au12", "au13", "au14", "au15", "au16", "au17", "au18", "au19",
                    "au20", "au22", "au23", "au24", "au25", "au26", "au27", "au32", "au38", "au39"]
    au_labels_27 = ["AU1", "AU2", "AU4", "AU5", "AU6", "AU7", "AU9", "AU10", "AU11",
     "AU12", "AU13", "AU14", "AU15", "AU16", "AU17", "AU18", "AU19",
     "AU20", "AU22", "AU23", "AU24", "AU25", "AU26", "AU27", "AU32", "AU38", "AU39"]
    targ = [0,1,2,4,6,9,21,22]
    # x = ["AU1", "AU2", "AU4", "AU6", "AU9","AU12", "AU25", "AU26",]
    auconf_27 = get_config_27()
    np.random.seed(1)
    auconf_27.evaluate = True
    auconf_27.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]

    set_env(auconf_27)
    AU_net_27 = MEFARG_27(num_main_classes=auconf_27.num_main_classes, num_sub_classes=auconf_27.num_sub_classes,
                          backbone=auconf_27.arc).to(device)
    AU_net_27 = load_state_dict(AU_net_27, auconf_27.resume).to(device)
    AU_net_27.eval()

    allVideos = sorted(glob.glob(args.inputpath))
    for videopath in allVideos:
            # savefolder =

            # k = j
            inputpath = glob.glob(os.path.join(videopath, '*.jpg'))

            # inputpath = args.inputpath.replace("id", actors[k]).replace("exp", expressionNames[j])
            # if not os.path.exists(inputpath):
            #     continue
            video_name = videopath.split('/')[-2]
            savefolder = args.savefolder.replace("name", video_name)
            # vidoname = args.vidoname.replace("id", actors[k]).replace("exp", expressionNames[j])
            if os.path.exists(savefolder):
                print("exists...")
                continue
            os.makedirs(savefolder, exist_ok=True)
            os.makedirs(os.path.join(savefolder, "AU"), exist_ok=True)
            testdata = datasets.TestData(inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                         scale=1.25, )

            color = np.random.uniform(0, 1, (27, 3))
            # os.makedirs(os.path.join(savefolder), exist_ok=True)
            # mmk = []

            for i in tqdm(range(1,len(testdata)-1)):
            # for i in tqdm(range(1,2)):
                data_1 = testdata[i - 1]
                data = testdata[i]
                # print((i + 1) % len(testdata))
                data_3 = testdata[(i + 1) % len(testdata)]

                name = data['imagename']

                images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                                   0).to(device)

                # fileName2 = os.path.join(savefolder, "au27", name.split("LeftVideo")[-1].split("_")[0])
                with torch.no_grad():
                    # print(datetime.now)
                    codedict_old, codedict = deca.encode(images)
                    opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor

                    # print(datetime.now)
                    if args.render_orig:
                        tform = testdata[i]['tform'][None, ...]
                        tform = torch.inverse(tform).transpose(1, 2).to(device)
                        original_image = testdata[i]['original_image'][None, ...].to(device)
                        _, orig_visdict = deca.decode(codedict, codedict_old, render_orig=True,
                                                      original_image=original_image, tform=tform)
                        orig_visdict['inputs'] = original_image
                # image_au = AU_net_27(images[1:2])
                rend_au = AU_net_27(opdict['rendered_images'])
                # rend_au_deca = AU_net_27(opdict['rendered_images_emoca'])
                # print(rend_au[1].float()[0])
                rects_2 = plt.barh(au_labels_27,
                                   rend_au[1].float()[0].cpu().detach().numpy()[:27],
                                   color=color)
                # rects_2 = plt.barh(x,
                #                    rend_au[1].float()[0].cpu().detach().numpy()[targ],
                #                    color=color)
                # print(rects_2)

                plt.xlim((0, 1))
                add_labels(rects_2)
                plt.savefig(os.path.join(savefolder, "AU",name + '_AU.jpg'))

                # plt.show()
                plt.clf()

                _, input_exp = expression_net.forward2(images[1:2])
                _, rend_exp_pre = expression_net.forward2(opdict['rendered_images'])
                _, rend_exp_pre_emo = expression_net.forward2(opdict['rendered_images_emoca'])
                # _, input_exp = expression_net(images[1:2])
                # _, rend_exp_pre = expression_net(opdict['rendered_images'])
                # _, rend_exp_pre_emo = expression_net(opdict['rendered_images_emoca'])
                # print(name,input_exp,rend_exp_pre,rend_exp_pre_emo )
                nXM = torch.nn.Softmax(dim=1)
                # print(name, nXM(input_exp), nXM(rend_exp_pre), nXM(rend_exp_pre_emo))
                # print(name, torch.argmax(nXM(input_exp), dim=1), torch.argmax(nXM(rend_exp_pre), dim=1),
                #       torch.argmax(nXM(rend_exp_pre_emo), dim=1))
                # print(nXM(input_exp).cpu().detach().numpy()[:8])
                # plt.bar(range(8), nXM(input_exp).cpu().detach().numpy()[0][:8], color=['r','g','b',[0.3010,0.7450,0.9330],'m','y',[0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880]])
                # rects_1= plt.barh(['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry','Contempt'], nXM(input_exp).cpu().detach().numpy()[0][:8], color=['r','g','b',[0.3010,0.7450,0.9330],'m','y',[0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880]])
                # plt.xlim((0,1))
                # add_labels(rects_1)
                # plt.show()
                rects_2 = plt.barh(['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry','Contempt'], nXM(rend_exp_pre).cpu().detach().numpy()[0][:8], color=['r','g','b',[0.3010,0.7450,0.9330],'m','y',[0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880]])
                # print(rects_2)

                plt.xlim((0,1))
                add_labels(rects_2)
                plt.savefig(os.path.join(savefolder,  "AU", name + '_Emotion.jpg'))
                plt.clf()
                # plt.show()
                # cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)

                # rects_3 = plt.barh(['Neutral','Happy','Sad','Surprise','Fear','Disgust','Angry','Contempt'], nXM(rend_exp_pre_emo).cpu().detach().numpy()[0][:8], color=['r','g','b',[0.3010,0.7450,0.9330],'m','y',[0.4940,0.1840,0.5560],[0.4660,0.6740,0.1880]])
                # plt.xlim((0,1))
                # add_labels(rects_3)
                # plt.show()
                # print(rects_1, rects_2, rects_3)
                # print(rects_2)

                vis_image = deca.visualize(visdict, size=448)
                # v = cv2.vconcat([vis_image[:,:448],vis_image[:,448*4:448*5],vis_image[:,448*6:],cv2.resize(cv2.imread(os.path.join(savefolder, 'result', name + '_AU.jpg'))[20:-20,20:-20],[448, 448], interpolation=cv2.INTER_NEAREST)])
                v = cv2.vconcat([cv2.resize(vis_image[:,:448],[640,480], interpolation=cv2.INTER_AREA),cv2.resize(vis_image[:,448*4:448*5],[640,480], interpolation=cv2.INTER_AREA),cv2.imread(os.path.join(savefolder, "AU",name + '_AU.jpg')), cv2.imread(os.path.join(savefolder, "AU",name + '_Emotion.jpg'))])
                cv2.imwrite(os.path.join(savefolder,  name + '.jpg'), v)
                # cv2.imwrite(os.path.join(savefolder,   'Atemp.jpg'), v)
                # if i==1:
                #     mmk = v
                # else:
                #     mmk = cv2.hconcat([mmk, v])
                # cv2.imwrite(os.path.join(savefolder,  'Atemp2.jpg'),mmk)

            print(f'-- please check the results in {savefolder}')

    # print(savefolder, savefolder.split('/')[-2]+'.jpg')
    # cv2.imwrite(os.path.join(savefolder, savefolder.split('/')[-2]+'.jpg'),mmk)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry2_same/",
    # patr = "Left13/1102_1110/"
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/"+patr,
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/croppedImages1_1_2/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/home/cine/Downloads/DFEW/DFEW-part2/Clip/clip_224x224/clip_224/*/",

    #                     parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry_sameF/",
                        # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry2_same/",
                        # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry3_same/",
                        # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry3_same/",
                        # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/Actor1_angry/",
                        # parser.add_argument('-i', '--inputpath', default="/home/cine/Desktop/ForSeuqenceFrames/differentID/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        default='/media/cine/First/Results/Sequence_AUPrediction/DFEW2/name/',

                        # default='/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain6/ForCheck_DISFA/'+patr,
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
