"""
RAVDESS 데이터셋을 모두 처리하는 파일.
demo_testAU_OnlyE_ver1.py의 전체 데이터셋 처리버전
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
from decalib.gatfarec_Video_DetailNew_20260104_checkRendering import DECA
from decalib.datasets import datasets2 as datasets
from decalib.utils import util
from decalib.utils.config_wt_DetailNew_20260103 import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points


def main(args):
    # if args.rasterizer_type != 'standard':
    #     args.render_orig = False
    base_savefolder = args.savefolder
    device = args.device
    
    # Get all subdirectories under the input path
    base_inputpath = args.inputpath.rstrip('/')  # Remove trailing slash for consistency
    
    if not os.path.isdir(base_inputpath):
        print(f"Error: Input path {base_inputpath} does not exist!")
        return
    
    # Get all subdirectories under the input path
    subdirs = [d for d in os.listdir(base_inputpath) 
               if os.path.isdir(os.path.join(base_inputpath, d))]
    subdirs = sorted(subdirs)  # Sort for consistent processing order
    
    if len(subdirs) == 0:
        print(f"Warning: No subdirectories found in {base_inputpath}")
        return
    
    print(f"Found {len(subdirs)} directories to process: {subdirs}")
    
    # Process each directory
    for dir_name in subdirs:
        inputpath = os.path.join(base_inputpath, dir_name)
        savefolder = os.path.join(base_savefolder, dir_name)
        os.makedirs(savefolder, exist_ok=True)
        
        print(f"\nProcessing directory: {dir_name}")
        print(f"Input path: {inputpath}")
        print(f"Save folder: {savefolder}")

        # load test images
        # testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)
        testdata = datasets.TestData(inputpath, iscrop=args.iscrop, crop_size=deca_cfg.dataset.image_size,
                                     scale=1.1, )
        
        if len(testdata) < 3:
            print(f"Warning: Directory {dir_name} has less than 3 images, skipping...")
            continue
            
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # 3행 3열 레이아웃: 각 이미지 448x448, 가로 3개, 세로 3개
        vidoname = dir_name if args.vidoname == 'Actor_03sad' else args.vidoname
        out = cv2.VideoWriter(os.path.join(savefolder, vidoname + ".mp4"), fourcc, 30, (448 * 3, 448 * 3), True)
        # run DECA
        deca_cfg.model.use_tex = args.useTex
        deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
        # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
        deca_cfg.rasterizer_type = args.rasterizer_type
        deca_cfg.model.extract_tex = args.extractTex
        deca = DECA(config=deca_cfg, device=device)
        # 결과 이미지 디렉터리에 로그 파일 생성 설정
        deca.verification_log_path = os.path.join(savefolder, 'rendering_verification.log')
        # uv_z를 0으로 설정하는 옵션 전달
        deca.zero_uv_z = args.zero_uv_z
        # DECA 알베도 사용 옵션 전달
        deca.use_deca_albedo = args.use_deca_albedo
        
        if args.zero_uv_z:
            print("!!! Verification Mode: uv_z is forced to 0.0 !!!")
        if args.use_deca_albedo:
            print("!!! Professor's Experiment: Using DECA Albedo for our rendering !!!")
        # writer = pd.ExcelWriter(
        #         os.path.join(savefolder, 'parameters.xlsx'))
        # # for i in range(len(testdata)):
        # writeContent = []

        os.makedirs(os.path.join(savefolder, 'result'), exist_ok=True)
        for i in tqdm(range(1, len(testdata)-1), desc=f"Processing {dir_name}"):
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]
            name = data['imagename']

            images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]), 0).to(
                device)

            # data = testdata[i]
            # name = data['imagename']

            # images = testdata[i]['image'].to(device)[None,...]

            with torch.no_grad():
                codedict_old, codedict = deca.encode(images )
                # codedict['pose'][0][:3] = 0.
                # codedict['cam'] = codedict_deca['cam']
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=True)  # tensor

                # if args.render_orig:
                #     tform = testdata[i]['tform'][None, ...]
                #     tform = torch.inverse(tform).transpose(1,2).to(device)
                #     original_image = testdata[i]['original_image'][None, ...].to(device)
                #     _, orig_visdict = deca.decode(codedict, codedict_decarender_orig=True, original_image=original_image, tform=tform)
                #     orig_visdict['inputs'] = original_image
            # if i ==0:
            #     writeContent = opdict['parameters'].to('cpu').detach()
            #     writeContent_deca = opdict['parameters_deca'].to('cpu').detach()
            # else:
            #     writeContent = torch.cat((writeContent, opdict['parameters'].to('cpu').detach()), dim=0)
            #     writeContent_deca = torch.cat((writeContent_deca, opdict['parameters_deca'].to('cpu').detach()), dim=0)

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
                np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
            if args.saveObj:
                # deca.save_obj(os.path.join(savefolder, 'obj', name + '.obj'), opdict)
                landmark_51 = opdict['landmarks3d_world'][:, 17:]
                landmark_7 = landmark_51[:, [19, 22, 25, 28, 16, 31, 37]]
                landmark_7 = landmark_7.cpu().numpy()
                np.save(os.path.join(savefolder, 'landmark_7', name + '.npy'), landmark_7[0])

            # cv2.imwrite(os.path.join(savefolder, 'result',name + '.jpg'), deca.visualize(visdict))
            vis_image = deca.visualize(visdict, size=448)
            # orig_vis_image = deca.visualize(orig_visdict, size=448)

            cv2.imwrite(os.path.join(savefolder, 'result', name + '.jpg'), vis_image)
            # cv2.imwrite(os.path.join(savefolder, 'result_original', name + '.jpg'), orig_vis_image)
            out.write(vis_image)
            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
            if args.saveVis:
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
                # if args.render_orig:
                #     cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
            if args.saveImages:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)
                for vis_name in ['uv_albedo', 'uv_normal_coarse', 'uv_normal_detail', 'uv_displacement']:
                    if vis_name in visdict:
                        img = util.tensor2image(visdict[vis_name][0])
                        cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'), img)
            
            # cv2.imwrite(os.path.join(savefolder, 'result',name + '.jpg'), deca.visualize(visdict))
    
    print(f'\nAll directories processed. Results saved in {base_savefolder}')
    # data1 = pd.DataFrame(np.array(writeContent, dtype=np.float32))
    # data2 = pd.DataFrame(np.array(writeContent_deca, dtype=np.float32))

    # data1.to_excel(writer, 'data', float_format='%.5f')
    # data2.to_excel(writer, 'deca', float_format='%.5f')
    # print('please check parameter in parameters.xlsx')
    # writer.save()
    # writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')
    # parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
    # # 08, 30, 37, 43, 64, 75
    name = 'Actor_03sad'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # name = '002645310-'  # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # sentence = 'SEN_do_they_make_class_biased_decisions'
    # name+=sentence
    # name ='spectre_video' # angry, calm, disgust, fear, happy, neutr, sad, surprise
    # name ='35-30-1920x1080_68'
    # parser.add_argument('-i', '--inputpath', default='/media/cine/First/TestDataset/images/FaMoS_subject_064/', type=str,
    parser.add_argument('-vn', '--vidoname', default=name, type=str, )  # # 05happy 14  16calm  18disgust 18sad
    # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/actors/' + name, type=str,
    # Changed to process all directories under /home/cine/Downloads/actors/images/
    parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/actors/images/', type=str,
    # parser.add_argument('-i', '--inputpath', default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI_7_17/multiface/mini_datasetB/m--20190828--1318--002645310--GHS/images2/'+sentence+'/*', type=str,
                        # 05happy 14  16calm  18disgust 18sad
                        # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/spectre_video/images/', type=str, # 05happy 14  16calm  18disgust 18sad
                        # parser.add_argument('-i', '--inputpath', default='/home/cine/Downloads/woman_SR2/images/', type=str,
                        # parser.add_argument('-i', '--inputpath', default='/media/cine/First/Aff-wild2/images/video6_sequence/315/', type=str,
                        # parser.add_argument('-i', '--inputpath', default='/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/68/', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    # parser.add_argument('-s', '--savefolder', default='TestReult/pretrainNewIdea_236B/wild2/', type=str,
    # parser.add_argument('-s', '--savefolder', default='TestReult/pretrainNewIdea_236B/woman_SR2/', type=str,
    parser.add_argument('-s', '--savefolder', default='/media/cine/First/HWPJ2/ProjectResult/Demos/DetailNew_FT2_Demo/' + name,
                        type=str,
                        # parser.add_argument('-s', '--savefolder', default='TestReult/pretrainNewIdea_236B_/35-30-1920x1080_sequence/68X1_M', type=str,
                        # parser.add_argument('-s', '--savefolder', default='TestReult/pretrainNewIdea_236B/Actor_02angry_retina_e10/', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    # parser.add_argument('--pretrained_modelpath_ViT', default='/media/cine/First/TransformerCode/NewIdea3ViT/Training/pretrainA_150/model.tar'
    # parser.add_argument('--pretrained_modelpath_ViT', default='/media/cine/First/TransformerCode/NewIdea3ViT2/Training/pretrainB_236_2/model.tar', type=str, help='model.tar path')
    # parser.add_argument('--pretrained_modelpath_ViT', default='/home/cine/Documents/NewDECA_ViT_Video/Training/pretrain3_exp/model.tar', type=str, help='model.tar path')
    parser.add_argument('--pretrained_modelpath_ViT',
                        default='/media/cine/First/HWPJ2/ProjectResult/DetailNew_FineTune_2/model.tar', type=str,
                        help='model.tar path')
    # CUDA_VISIBLE_DEVICES가 설정되어 있으면 cuda:0, 없으면 cuda:1을 기본값으로 사용
    default_device = 'cuda:0' if os.environ.get('CUDA_VISIBLE_DEVICES') is not None else 'cuda:1'
    parser.add_argument('--device', default=default_device, type=str,
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
    parser.add_argument('--zero_uv_z', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='force uv_z to zero for identity verification')
    parser.add_argument('--use_deca_albedo', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='use DECA albedo for our model rendering')
    main(parser.parse_args())
