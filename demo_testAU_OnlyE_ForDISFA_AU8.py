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



from decalib.models.OpenGraphAU.model.MEFL import MEFARG as MEFARG_27
# from decalib.models.OpenGraphAU.utils import load_state_dict as
from decalib.models.OpenGraphAU.conf import get_config as get_config_27
# from utils_MG import statistics, update_statistics_list,
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

    # au_labels = ["au1", "au2", "au4", "au5", "au6", "au9",
    #              "au12", "au15", "au17", "au20", "au25", "au26"]
    au_labels = ["au1", "au2", "au4", "au6", "au9",
                 "au12",  "au25", "au26"]

    au_labels_27 = ["au1", "au2", "au4", "au5", "au6","au7", "au9","au10","au11",
                 "au12", "au13","au14","au15","au16", "au17", "au18","au19",
                 "au20","au22","au23","au24", "au25", "au26","au27","au32","au38","au39"]
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    auconf = get_config()

    auconf.evaluate = True
    auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    set_env(auconf)
    AU_net = MEFARG(num_classes=auconf.num_classes,
                         backbone=auconf.arc).to(device)
    AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    AU_net.eval()

    auconf_27 = get_config_27()

    auconf_27.evaluate = True
    auconf_27.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]

    set_env(auconf_27)
    AU_net_27 = MEFARG_27(num_main_classes=auconf_27.num_main_classes, num_sub_classes=auconf_27.num_sub_classes,
                         backbone=auconf_27.arc).to(device)
    AU_net_27 = load_state_dict(AU_net_27, auconf_27.resume).to(device)
    AU_net_27.eval()
    allVideos = sorted(glob.glob(args.inputpath), reverse=False)
    for videopath in allVideos:
        # for k in range(len(actors)):
            # savefolder =
            inputpath = videopath
            name = videopath.split("/croppedImages2")[0].split("/")[-1]
            # name = videopath.split("video_EMOCA/")[-1].split("/")[0]
            savefolder = args.savefolder.replace("*",name)
            # if os.path.exists(savefolder):
            #     continue
            fileName1 = os.path.join(savefolder, "au8",name.split("LeftVideo")[-1].split("_")[0])
            fileName2 = os.path.join(savefolder, "au27",name.split("LeftVideo")[-1].split("_")[0])

            # fileName1 = os.path.join(savefolder, "au1",name.split("LeftVideo")[-1].split("_")[0])
            # fileName2 = os.path.join(savefolder, "au2",name.split("LeftVideo")[-1].split("_")[0])
            # vidoname = args.vidoname.replace("*",name)
            # if os.path.exists(savefolder):
            #     print("exists...")
            #     continue
            os.makedirs(savefolder, exist_ok=True)
            if os.path.exists(os.path.join(savefolder, 'au8')):
                print(savefolder)
                continue


            # load test images
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

            os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, 'au8'), exist_ok=True)
            os.makedirs(os.path.join(savefolder, 'au27'), exist_ok=True)
            # os.makedirs(os.path.join(savefolder, 'au1'), exist_ok=True)
            # os.makedirs(os.path.join(savefolder, 'au2'), exist_ok=True)
            os.makedirs(os.path.splitext(inputpath)[0].replace("originalImages", "croppedImages"),exist_ok=True )

            for i in tqdm(range(0, len(testdata) - 1)):
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

                image_au,_ = AU_net(images[1:2])
                rend_au,_ = AU_net(opdict['rendered_images'])
                rend_au_deca,_ = AU_net(opdict['rendered_images_emoca'])

                image_au = image_au >= 0.5
                rend_au = rend_au >= 0.5
                rend_au_deca = rend_au_deca >= 0.5
                image_au = image_au.long()
                rend_au = rend_au.long()
                rend_au_deca = rend_au_deca.long()
                opdict['au_img'] = image_au[0]
                opdict['au_rend'] = rend_au[0]
                opdict['au_rend_deca'] = rend_au_deca[0]
                # resultI, resultR = vis_au(opdict['au_img'] , opdict['au_rend'] )
                for kt, xt in enumerate(au_labels):
                    # print(opdict['au_img'][kt], opdict['au_rend'][kt])
                    with open(fileName1+"_"+xt+".txt", "a") as f:
                        f.write(str(opdict['au_img'].cpu().numpy().tolist()[kt])+"\n")
                    with open(fileName1+"_"+xt+"R.txt", "a") as f:
                        f.write(str(opdict['au_rend'].cpu().numpy().tolist()[kt])+"\n")
                    with open(fileName1+"_"+xt+"R_emoca.txt", "a") as f:
                        f.write(str(opdict['au_rend_deca'].cpu().numpy().tolist()[kt])+"\n")

                image_au = AU_net_27(images[1:2])
                rend_au = AU_net_27(opdict['rendered_images'])
                rend_au_deca = AU_net_27(opdict['rendered_images_emoca'])


                opdict['au_img'] = (image_au[1] >= 0.5).float()[0]
                opdict['au_rend'] = (rend_au[1] >= 0.5).float()[0]
                opdict['au_rend_deca'] = (rend_au_deca[1] >= 0.5).float()[0]
                # resultI, resultR = vis_au(opdict['au_img'] , opdict['au_rend'] )
                for kt, xt in enumerate(au_labels_27):
                    # print(opdict['au_img'][kt], opdict['au_rend'][kt])
                    with open(fileName2+"_"+xt+".txt", "a") as f:
                        f.write(str(opdict['au_img'].cpu().numpy().tolist()[kt])+"\n")
                    with open(fileName2+"_"+xt+"R.txt", "a") as f:
                        f.write(str(opdict['au_rend'].cpu().numpy().tolist()[kt])+"\n")
                    with open(fileName2+"_"+xt+"R_emoca.txt", "a") as f:
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

                # cv2.imwrite(os.path.join(savefolder, 'result',name + '.jpg'), deca.visualize(visdict))
                vis_image = deca.visualize(visdict, size=448)
                # orig_vis_image = deca.visualize(orig_visdict, size=448)

                cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # neural, calm, happy, sad, angry, fearful, disgust, surprised
    # name = 'Actor_01/calm'  # (02) angry, (x) calm, (10) disgust, (x) fear, (14)happy, neutr, (3) sad, (18)18surprise
    parser.add_argument('-vn', '--vidoname', default="*",
                        type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default="/home/cine/Documents/RADESS/originalImages/Actor_id/exp",
    parser.add_argument('-i', '--inputpath', default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video/*/croppedImages2/",
    # parser.add_argument('-i', '--inputpath', default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video_EMOCA/*/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain2/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_CropBEmoca/AULoss1_ELT/*/',
                        default='/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain6/DISFA/*/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_CropBEmoca/pretrain5X_25/*/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_2/pretrain4/*/',
                        type=str, help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--pretrained_modelpath_ViT',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1/models/00820199.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X/model.tar',
                        # default='/home/cine/Documents/HJCode/AU_sequence/Training1_videoC_OnlyE/pretrain1X1/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain2/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain4/model.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE/pretrain5X/models/00410099.tar',
                        # default='/media/cine/First/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain1/models/00410099.tar',
                        default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain6/model.tar',
                        # default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss/pretrain6/models/00131231.tar',

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
