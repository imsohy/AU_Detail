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
from decalib.models.OpenGraphAU.conf_DISFA import get_config, set_env

from decalib.models.OpenGraphAU.model.MEFL import MEFARG as MEFARG_27
# from decalib.models.OpenGraphAU.utils import load_state_dict as
from decalib.models.OpenGraphAU.conf import get_config as get_config_27


# from utils_MG import statistics, update_statistics_list,


# return sAU, dAU

def main(args):
    # au_labels = ["au1", "au2", "au4", "au6", "au9",
    #              "au12",  "au25", "au26"]

    au_labels_27 = ["au1", "au2", "au4", "au5", "au6", "au7", "au9", "au10", "au11",
                    "au12", "au13", "au14", "au15", "au16", "au17", "au18", "au19",
                    "au20", "au22", "au23", "au24", "au25", "au26", "au27", "au32", "au38", "au39"]
    device = args.device
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.pretrained_modelpath = args.pretrained_modelpath_ViT
    # deca_cfg.model_path_HJ = '/home/cine/Documents/HJCode/GANE_code/Training/testGATE30/model.tar'
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config=deca_cfg, device=device)
    # auconf = get_config()
    #
    # auconf.evaluate = True
    # auconf.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]
    # # set_env(auconf)
    # AU_net = MEFARG(num_classes=auconf.num_classes,
    #                      backbone=auconf.arc).to(device)
    # AU_net = load_state_dict(AU_net, auconf.resume).to(device)
    # AU_net.eval()

    auconf_27 = get_config_27()

    auconf_27.evaluate = True
    auconf_27.gpu_ids = os.environ["CUDA_VISIBLE_DEVICES"]

    set_env(auconf_27)
    AU_net_27 = MEFARG_27(num_main_classes=auconf_27.num_main_classes, num_sub_classes=auconf_27.num_sub_classes,
                          backbone=auconf_27.arc).to(device)
    AU_net_27 = load_state_dict(AU_net_27, auconf_27.resume).to(device)
    AU_net_27.eval()
    # allVideos = sorted(glob.glob(args.inputpath), reverse=False)
    allVideos = ['/media/cine/First/Aff-wild2/images/84-30-1920x1080_sequence/43/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/137/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/538/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/137/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/116/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/12/',
                 '/media/cine/First/Aff-wild2/images/video95_sequence/41/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/238/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/120/',
                 '/media/cine/First/Aff-wild2/images/131-30-1920x1080_sequence/401/',
                 '/media/cine/First/Aff-wild2/images/122-60-1920x1080-1_sequence/30/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/282/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/330/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/6/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/45/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/670/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/103/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/31/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/253/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/121/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/99/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/107/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/42/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/87/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/39/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/33/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/121-24-1920x1080_sequence/53/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/107/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/124/',
                 '/media/cine/First/Aff-wild2/images/87-25-1920x1080_sequence/42/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/57/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/301/',
                 '/media/cine/First/Aff-wild2/images/video86_3_sequence/2/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/39/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/48/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/311/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/33/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/142/',
                 '/media/cine/First/Aff-wild2/images/video34_sequence/6/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/808/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/194/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/68/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/324/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1016/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/226/',
                 '/media/cine/First/Aff-wild2/images/111-25-1920x1080_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/103/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-4_sequence/61/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/82/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/228/',
                 '/media/cine/First/Aff-wild2/images/video7_sequence/1079/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/68/',
                 '/media/cine/First/Aff-wild2/images/114-30-1280x720_sequence/10/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/174/',
                 '/media/cine/First/Aff-wild2/images/video39_sequence/127/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1109/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1495/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/299/',
                 '/media/cine/First/Aff-wild2/images/video39_sequence/140/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/357/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/111/',
                 '/media/cine/First/Aff-wild2/images/87-25-1920x1080_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/34/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/865/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/53/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/717/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/404/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/167/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/73/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/219/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/302/',
                 '/media/cine/First/Aff-wild2/images/77-30-1280x720_sequence/14/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/111/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/427/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/796/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/33/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/505/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/94/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/144/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/174/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/148/',
                 '/media/cine/First/Aff-wild2/images/video25_sequence/89/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/98/',
                 '/media/cine/First/Aff-wild2/images/60-30-1920x1080_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/48/',
                 '/media/cine/First/Aff-wild2/images/84-30-1920x1080_sequence/28/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/103/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/109/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/189/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/137/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/228/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/341/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/23/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/232/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/600/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/197/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/164/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/129/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/71/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1057/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/201/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/357/',
                 '/media/cine/First/Aff-wild2/images/95-24-1920x1080_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/27/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/3/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/38/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/135/',
                 '/media/cine/First/Aff-wild2/images/60-30-1920x1080_sequence/31/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/163/',
                 '/media/cine/First/Aff-wild2/images/87-25-1920x1080_sequence/17/',
                 '/media/cine/First/Aff-wild2/images/video47_sequence/4/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/528/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/423/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/48/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/239/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/569/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/305/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/135/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/80/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/230/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/786/',
                 '/media/cine/First/Aff-wild2/images/131-30-1920x1080_sequence/382/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/51/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/51/',
                 '/media/cine/First/Aff-wild2/images/54-30-1080x1920_sequence/42/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/60/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/394/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/56/',
                 '/media/cine/First/Aff-wild2/images/83-24-1920x1080_sequence/564/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/368/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/186/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1412/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1685/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/43/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/891/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/11/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/549/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/78/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/161/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/881/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/116/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/52/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/109/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/18/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/299/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/129/',
                 '/media/cine/First/Aff-wild2/images/134-30-1280x720_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/8/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/139/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/157/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/505/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/308/',
                 '/media/cine/First/Aff-wild2/images/14-30-1920x1080_sequence/197/',
                 '/media/cine/First/Aff-wild2/images/14-30-1920x1080_sequence/261/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/824/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1282/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/90/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/427/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/313/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/173/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1301/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/180/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/16/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-1_sequence/127/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/226/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1265/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/347/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/235/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/37/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/641/',
                 '/media/cine/First/Aff-wild2/images/100-29-1080x1920_sequence/80/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/66/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/107/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/8/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/67/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/136/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/136-30-1920x1080_sequence/2166/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/110/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/24/',
                 '/media/cine/First/Aff-wild2/images/12-24-1920x1080_sequence/154/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/594/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/94/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/705/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/21/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/351/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/54/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1410/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/165/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/123/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-2_sequence/33/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/380/',
                 '/media/cine/First/Aff-wild2/images/video78_sequence/10/',
                 '/media/cine/First/Aff-wild2/images/83-24-1920x1080_sequence/534/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/67/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/video96_sequence/23/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/49/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1013/',
                 '/media/cine/First/Aff-wild2/images/136-30-1920x1080_sequence/18/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/304/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/186/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/587/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/215/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/347/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/96/',
                 '/media/cine/First/Aff-wild2/images/video33_sequence/518/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/255/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/40/',
                 '/media/cine/First/Aff-wild2/images/video37_sequence/20/',
                 '/media/cine/First/Aff-wild2/images/video51_sequence/129/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/251/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/359/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/903/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/144/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/222/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/38/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/56/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/339/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/136/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/11/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/261/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/776/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/142/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/279/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/190/',
                 '/media/cine/First/Aff-wild2/images/32-60-1920x1080_sequence/278/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/209/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/162/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/583/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/177/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/131/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/199/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/43/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/144/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/270/',
                 '/media/cine/First/Aff-wild2/images/77-30-1280x720_sequence/8/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/464/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/165/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/35/',
                 '/media/cine/First/Aff-wild2/images/video95_sequence/360/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/395/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/536/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/793/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/122/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/50/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/427/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/161/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/91/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/217/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/15/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/250/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/141/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/173/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/46/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/20/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/195/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/260/',
                 '/media/cine/First/Aff-wild2/images/video6_sequence/49/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/129/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-2_sequence/78/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/218/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/109/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/722/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-1_sequence/21/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/385/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/51/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/150/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/326/',
                 '/media/cine/First/Aff-wild2/images/124-30-720x1280_sequence/354/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/25/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/118/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/247/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/358/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1421/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/173/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/81/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/557/',
                 '/media/cine/First/Aff-wild2/images/83-24-1920x1080_sequence/176/',
                 '/media/cine/First/Aff-wild2/images/video67_sequence/55/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/42/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/55/',
                 '/media/cine/First/Aff-wild2/images/54-30-1080x1920_sequence/27/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1190/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/582/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/385/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/111/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/289/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/video39_sequence/80/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/149/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1252/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/110/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/50/',
                 '/media/cine/First/Aff-wild2/images/131-30-1920x1080_sequence/359/',
                 '/media/cine/First/Aff-wild2/images/video25_sequence/205/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1136/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/473/',
                 '/media/cine/First/Aff-wild2/images/video94_sequence/14/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/109/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/59/',
                 '/media/cine/First/Aff-wild2/images/83-24-1920x1080_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/video25_sequence/10/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/894/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/140/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/279/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/269/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/34/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/61/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/112/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/405/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/193/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/129/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/174/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/172/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/258/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/98/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/266/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/83/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/86/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/70/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/4/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/861/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/60/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/75/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/14/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/28/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/48/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/37/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/123/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/7/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/196/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/239/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/416/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/802/',
                 '/media/cine/First/Aff-wild2/images/84-30-1920x1080_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/383/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/455/',
                 '/media/cine/First/Aff-wild2/images/video51_sequence/31/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1731/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/60/',
                 '/media/cine/First/Aff-wild2/images/video51_sequence/115/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/80/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/4/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/37/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/39/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/110/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/55/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/125/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1282/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/117/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/298/',
                 '/media/cine/First/Aff-wild2/images/87-25-1920x1080_sequence/11/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/344/',
                 '/media/cine/First/Aff-wild2/images/100-29-1080x1920_sequence/83/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/168/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/282/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/9/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/71/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/152/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/105/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/79/',
                 '/media/cine/First/Aff-wild2/images/49-30-1280x720_sequence/40/',
                 '/media/cine/First/Aff-wild2/images/video78_sequence/13/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/178/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/69/',
                 '/media/cine/First/Aff-wild2/images/video34_sequence/59/',
                 '/media/cine/First/Aff-wild2/images/111-25-1920x1080_sequence/3/',
                 '/media/cine/First/Aff-wild2/images/video51_sequence/112/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/110/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/53/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/97/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/59/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/70/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/329/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/715/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/86/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/20/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1389/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/356/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/222/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/366/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/4/',
                 '/media/cine/First/Aff-wild2/images/77-30-1280x720_sequence/21/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-1_sequence/19/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/234/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/88/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1309/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1716/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/344/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/31/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/104/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/83/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/285/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/490/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/110/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/69/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/36/',
                 '/media/cine/First/Aff-wild2/images/video34_sequence/19/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/220/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/318/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/112/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/750/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/46/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/20/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/61/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/40/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/126/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/65/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/18/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/201/',
                 '/media/cine/First/Aff-wild2/images/100-29-1080x1920_sequence/112/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/73/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/125/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/48/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/104/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/263/',
                 '/media/cine/First/Aff-wild2/images/50-30-1920x1080_sequence/84/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/17/',
                 '/media/cine/First/Aff-wild2/images/77-30-1280x720_sequence/9/',
                 '/media/cine/First/Aff-wild2/images/12-24-1920x1080_sequence/12/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/101/',
                 '/media/cine/First/Aff-wild2/images/136-30-1920x1080_sequence/2172/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/167/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/446/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/860/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/11/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/77/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/444/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/987/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/257/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1083/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/42/',
                 '/media/cine/First/Aff-wild2/images/100-29-1080x1920_sequence/105/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1198/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/282/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/256/',
                 '/media/cine/First/Aff-wild2/images/video86_2_sequence/6/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/71/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/82/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/424/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/198/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/28/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/114/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1657/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/173/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/70/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/142/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/89/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/2/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/9/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/20/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/278/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/78/',
                 '/media/cine/First/Aff-wild2/images/video89_sequence/177/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/67/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/61/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/28/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/175/',
                 '/media/cine/First/Aff-wild2/images/84-30-1920x1080_sequence/19/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/15/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/252/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/628/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/485/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/17/',
                 '/media/cine/First/Aff-wild2/images/66-25-1080x1920_sequence/99/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/463/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/95/',
                 '/media/cine/First/Aff-wild2/images/14-30-1920x1080_sequence/36/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/95/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-2_sequence/76/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/19/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1703/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/31/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/14/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/210/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/16/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/49/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/1015/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/241/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/44/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/25/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/123/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/641/',
                 '/media/cine/First/Aff-wild2/images/video2_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/75/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1633/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/29/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/172/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/76/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/11/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/103/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/151/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/78/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/124/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/278/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/187/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/6/',
                 '/media/cine/First/Aff-wild2/images/video33_sequence/510/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/35/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/47/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/122-60-1920x1080-4_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/456/',
                 '/media/cine/First/Aff-wild2/images/60-30-1920x1080_sequence/47/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/9/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/video24_sequence/92/',
                 '/media/cine/First/Aff-wild2/images/100-29-1080x1920_sequence/3/',
                 '/media/cine/First/Aff-wild2/images/video34_sequence/53/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/460/',
                 '/media/cine/First/Aff-wild2/images/70-30-720x1280_sequence/24/',
                 '/media/cine/First/Aff-wild2/images/video37_sequence/3/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/236/',
                 '/media/cine/First/Aff-wild2/images/14-30-1920x1080_sequence/200/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/322/',
                 '/media/cine/First/Aff-wild2/images/83-24-1920x1080_sequence/592/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/254/',
                 '/media/cine/First/Aff-wild2/images/122-60-1920x1080-2_sequence/64/',
                 '/media/cine/First/Aff-wild2/images/87-25-1920x1080_sequence/12/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/3/',
                 '/media/cine/First/Aff-wild2/images/49-30-1280x720_sequence/84/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/28/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/90/',
                 '/media/cine/First/Aff-wild2/images/video15_sequence/147/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/256/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/87/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/292/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/249/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1187/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/10/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/47/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/243/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/72/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/195/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/351/',
                 '/media/cine/First/Aff-wild2/images/video47_sequence/8/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/315/',
                 '/media/cine/First/Aff-wild2/images/131-30-1920x1080_sequence/402/',
                 '/media/cine/First/Aff-wild2/images/video16_sequence/13/',
                 '/media/cine/First/Aff-wild2/images/95-24-1920x1080_sequence/43/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/399/',
                 '/media/cine/First/Aff-wild2/images/video39_sequence/79/',
                 '/media/cine/First/Aff-wild2/images/60-30-1920x1080_sequence/54/',
                 '/media/cine/First/Aff-wild2/images/video22_sequence/51/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/183/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/143/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/26/',
                 '/media/cine/First/Aff-wild2/images/video59_sequence/130/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/641/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/13/',
                 '/media/cine/First/Aff-wild2/images/video40_sequence/98/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/271/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/79/',
                 '/media/cine/First/Aff-wild2/images/14-30-1920x1080_sequence/262/',
                 '/media/cine/First/Aff-wild2/images/video67_sequence/91/',
                 '/media/cine/First/Aff-wild2/images/video44_sequence/107/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/675/',
                 '/media/cine/First/Aff-wild2/images/60-30-1920x1080_sequence/91/',
                 '/media/cine/First/Aff-wild2/images/136-30-1920x1080_sequence/2157/',
                 '/media/cine/First/Aff-wild2/images/4-30-1920x1080_sequence/121/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/586/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/407/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/5/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/74/',
                 '/media/cine/First/Aff-wild2/images/video16_sequence/97/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/138/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/403/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/108/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/50/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/246/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/190/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/12/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/503/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1139/',
                 '/media/cine/First/Aff-wild2/images/video95_sequence/234/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/16/',
                 '/media/cine/First/Aff-wild2/images/video95_sequence/147/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/332/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/181/',
                 '/media/cine/First/Aff-wild2/images/131-30-1920x1080_sequence/405/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/88/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/273/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/117/',
                 '/media/cine/First/Aff-wild2/images/video76_sequence/92/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/58/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1034/',
                 '/media/cine/First/Aff-wild2/images/video23_sequence/16/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/285/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/207/',
                 '/media/cine/First/Aff-wild2/images/137-30-1920x1080_sequence/309/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/386/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1722/',
                 '/media/cine/First/Aff-wild2/images/12-24-1920x1080_sequence/1/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/26/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/44/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/90/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/588/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/209/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-3_sequence/40/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/90/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/723/',
                 '/media/cine/First/Aff-wild2/images/video73_sequence/295/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/87/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/57/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/41/',
                 '/media/cine/First/Aff-wild2/images/video51_sequence/99/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/34/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/154/',
                 '/media/cine/First/Aff-wild2/images/video52_sequence/282/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/180/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/981/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/70/',
                 '/media/cine/First/Aff-wild2/images/16-30-1920x1080_sequence/21/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/57/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/77/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1521/',
                 '/media/cine/First/Aff-wild2/images/92-24-1920x1080_sequence/16/',
                 '/media/cine/First/Aff-wild2/images/74-25-1920x1080_sequence/7/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/64/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/948/',
                 '/media/cine/First/Aff-wild2/images/video13_sequence/199/',
                 '/media/cine/First/Aff-wild2/images/34-25-1920x1080_sequence/75/',
                 '/media/cine/First/Aff-wild2/images/90-30-1080x1920_sequence/225/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1075/',
                 '/media/cine/First/Aff-wild2/images/video95_sequence/355/',
                 '/media/cine/First/Aff-wild2/images/7-60-1920x1080_sequence/266/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/71/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/341/',
                 '/media/cine/First/Aff-wild2/images/12-24-1920x1080_sequence/22/',
                 '/media/cine/First/Aff-wild2/images/84-30-1920x1080_sequence/36/',
                 '/media/cine/First/Aff-wild2/images/video86_1_sequence/17/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/21/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/66/',
                 '/media/cine/First/Aff-wild2/images/video64_sequence/142/',
                 '/media/cine/First/Aff-wild2/images/68-24-1920x1080_sequence/188/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/168/',
                 '/media/cine/First/Aff-wild2/images/114-30-1280x720_sequence/2/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/290/',
                 '/media/cine/First/Aff-wild2/images/5-60-1920x1080-1_sequence/203/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/739/',
                 '/media/cine/First/Aff-wild2/images/35-30-1920x1080_sequence/248/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/278/',
                 '/media/cine/First/Aff-wild2/images/video82_sequence/547/',
                 '/media/cine/First/Aff-wild2/images/123-25-1920x1080_sequence/287/',
                 '/media/cine/First/Aff-wild2/images/video42_sequence/23/',
                 '/media/cine/First/Aff-wild2/images/9-15-1920x1080_sequence/241/',
                 '/media/cine/First/Aff-wild2/images/101-30-1080x1920_sequence/105/',
                 '/media/cine/First/Aff-wild2/images/video85_sequence/128/',
                 '/media/cine/First/Aff-wild2/images/24-30-1920x1080-2_sequence/41/',
                 '/media/cine/First/Aff-wild2/images/video27_sequence/622/',
                 '/media/cine/First/Aff-wild2/images/23-24-1920x1080_sequence/71/',
                 '/media/cine/First/Aff-wild2/images/video14_sequence/212/',
                 '/media/cine/First/Aff-wild2/images/video17_sequence/1394/',
                 '/media/cine/First/Aff-wild2/images/77-30-1280x720_sequence/15/',
                 '/media/cine/First/Aff-wild2/images/video69_sequence/1297/',
                 '/media/cine/First/Aff-wild2/images/97-29-1920x1080_sequence/238/']

    for videopath in allVideos:
        # for k in range(len(actors)):
        # savefolder =
        inputpath = videopath
        name = videopath.split("images/")[1]
        # name = videopath.split("video_EMOCA/")[-1].split("/")[0]
        savefolder = args.savefolder.replace("*", name)
        if os.path.exists(savefolder):
            continue
        fileName2 = os.path.join(savefolder, "au27", name.split("LeftVideo")[-1].split("_")[0])

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
        # os.makedirs(os.path.join(savefolder, 'result_original'), exist_ok=True)
        # os.makedirs(os.path.join(savefolder, 'au8'), exist_ok=True)
        os.makedirs(os.path.join(savefolder, 'au27'), exist_ok=True)
        # os.makedirs(os.path.join(savefolder, 'au1'), exist_ok=True)
        # os.makedirs(os.path.join(savefolder, 'au2'), exist_ok=True)
        # os.makedirs(os.path.splitext(inputpath)[0].replace("originalImages", "croppedImages"), exist_ok=True)

        for i in tqdm(range(0, len(testdata) - 1)):
            data_1 = testdata[i - 1]
            data = testdata[i]
            data_3 = testdata[i + 1]
            name = data['imagename']

            images = torch.cat((data_1['image'][None, ...], data['image'][None, ...], data_3['image'][None, ...]),
                               0).to(device)


            with torch.no_grad():
                codedict_old, codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict, codedict_old, use_detail=False)  # tensor




            image_au = AU_net_27(images[1:2])
            rend_au = AU_net_27(opdict['rendered_images'])
            rend_au_deca = AU_net_27(opdict['rendered_images_emoca'])

            opdict['au_img'] = (image_au[1] >= 0.5).float()[0]
            opdict['au_rend'] = (rend_au[1] >= 0.5).float()[0]
            opdict['au_rend_deca'] = (rend_au_deca[1] >= 0.5).float()[0]
            # resultI, resultR = vis_au(opdict['au_img'] , opdict['au_rend'] )
            for kt, xt in enumerate(au_labels_27):
                # print(opdict['au_img'][kt], opdict['au_rend'][kt])
                with open(fileName2 + "_" + xt + ".txt", "a") as f:
                    f.write(str(opdict['au_img'].cpu().numpy().tolist()[kt]) + "\n")
                with open(fileName2 + "_" + xt + "R.txt", "a") as f:
                    f.write(str(opdict['au_rend'].cpu().numpy().tolist()[kt]) + "\n")
                with open(fileName2 + "_" + xt + "R_emoca.txt", "a") as f:
                    f.write(str(opdict['au_rend_deca'].cpu().numpy().tolist()[kt]) + "\n")

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
    parser.add_argument('-i', '--inputpath',
                        default="*",
                        # parser.add_argument('-i', '--inputpath', default="/media/cine/de6afd1d-c444-4d43-a787-079519ace719/DISFA/video_EMOCA/*/",
                        type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain1X1/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/OnlyE/pretrain2/Actor_id/exp_texture/',
                        # default='/home/cine/Documents/ForPaperResult/TestReult/DISFA_CropBEmoca/AULoss1_ELT/*/',
                        default='/media/cine/de6afd1d-c444-4d43-a787-079519ace719/ForPaperResult/Aff-wild2_30/*',
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
                        default=  "/media/cine/de6afd1d-c444-4d43-a787-079519ace719/HJCode2/AUSequence/Training1_videoC_OnlyE_AULoss_7_3_2AFF/pretrain1/model.tar",
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
