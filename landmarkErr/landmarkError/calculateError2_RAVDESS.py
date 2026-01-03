import os.path

from tensor_cropper import transform_points, batch_kp_2d_l1_loss
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
# landmarkDir = sorted(glob("/media/cine/First/ForPaperResult/LandmarkVS/LandmarkVS/sequence_pretrain6_2/Actor_*/*/2d_landmark_68/*_FAN.npy"))
landmarkDir = sorted(glob("/media/cine/First/ForPaperResult/LandmarkVS/LandmarkVS/sequence_pretrain6_2/DFEW_/*/2d_landmark_68/*_FAN.npy"))
# landmarkDir_EMOCA = "/home/cine/Documents/ForPaperResult/TestReult/EMOCA/AFEW/*"
# landmarkDir_spectre = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*"
# landmarkDir_HJ = "/home/cine/Documents/ForPaperResult/TestReult/HJResult/AFEW/01/001/2d_landmark_68/00000.npy"
# landmarkDir_Seq = "/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain6/AFWE_VA/*"
# landmarkDir_Imag = "/media/cine/First/ForPaperResult/LandmarkVS/LandmarkVS/DECAbased/*_DECA.npy"
# landmarkDir_Imag = "/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/Image_pretrain4/AFWE_VA/*"
# /home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/01/001/lmk/frame0001.npy
# landmarkpathGT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy"
# landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"
# tformpath = "/home/cine/Downloads/AFEW-VA/tform/01/001/00001.npy"
EMOCA_based = 0.
DECA_based = 0.
# Aours_DECA = 0.
DECA = 0.
EMOCA = 0.
# Spectre = 0.
# HJImage = 0.
length = len(landmarkDir)
max1 = 0
min1 = 1000
max2 = 0
min2 = 1000
max3 = 0
min3 = 1000
max4 = 0
min4 = 1000
# max5 = 0
# min5 = 1000
# max6 = 0
# min6 = 1000
# max7 = 0
# min7 = 1000
for landmarkpath in tqdm(landmarkDir):
    # if "DECA" in landmarkpath:
    #     length-=1
    #     continue
    landmarks2dGT = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))[0]
    # name = os.path.splitext(os.path.split(landmarkpath)[-1])[0]
    #/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/01/001/2d_landmark_68/00001_DECA.npy
    # landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace("sequence_pretrain6_2","DECAbased").replace("_FAN","_DECA"), allow_pickle=True))
    landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace("sequence_pretrain6_2","DECAbased").replace("_FAN","_EMOCA"), allow_pickle=True))
    landmarkpath_DECA_based = torch.from_numpy(np.load(landmarkpath.replace("sequence_pretrain6_2","DECAbased").replace("_FAN",'_Ours'), allow_pickle=True))
    landmarkpath_EMOCA = torch.from_numpy(np.load(landmarkpath.replace("_FAN",'_EMOCA'), allow_pickle=True))
    landmarkpath_EMOCA_based =torch.from_numpy(np.load(landmarkpath.replace("_FAN",'_Ours'), allow_pickle=True))
    # landmarkpath_Spectre = landmarkDir_spectre.replace("*",landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/","lmk/"))
    # landmarkpathGT = landmarkDir_GT.replace("*", landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/",""))
    # if not os.path.exists(landmarkpath_Seq) or not os.path.exists(landmarkpath_Imag):
    #     continue
    # landmarks2d_Seq = torch.from_numpy(np.load(landmarkpath_Seq, allow_pickle=True))
    # landmarks2d_Imag = torch.from_numpy(np.load(landmarkpath_Imag, allow_pickle=True))
    # landmarks2d_EMOCA = torch.from_numpy(np.load(landmarkpath_EMOCA, allow_pickle=True))
    # landmarks2d_Spectre = torch.from_numpy(np.load(landmarkpath_Spectre, allow_pickle=True))
    # landmarks2dGT = torch.from_numpy(np.load(landmarkpathGT, allow_pickle=True) )

    loss1 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_DECA[17:])
    loss3 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarkpath_DECA_based[17:])
    loss2 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarkpath_EMOCA[17:])
    loss4 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarkpath_EMOCA_based[17:])
    # loss4 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_Spectre[17:])
    # loss5 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d[17:])
    if loss1 >0.5 or loss2>0.5 or loss3>0.5 or loss4>0.5:
        length-=1
        continue
    DECA += loss1
    DECA_based += loss3
    EMOCA += loss2
    EMOCA_based += loss4
    # HJImage += loss3
    # Spectre += loss4
    # Aours += loss5
    # Aours_Seq += loss6
    # Aours_Imag += loss7
    if loss1 < min1:
        min1 = loss1
    elif loss1 > max1:
        max1 = loss1
    if loss2 < min2:
        min2 = loss2
    elif loss2 > max2:
        max2 = loss2
    # if loss6 < min6:
    #     min6 = loss6
    # elif loss6 > max6:
    #     max6 = loss6
    # if loss7 < min7:
    #     min7 = loss7
    # elif loss7 > max7:
    #     max7 = loss7
    if loss4 < min4:
        min4 = loss4
    elif loss4 > max4:
        max4 = loss4
    if loss3 < min3:
        min3 = loss3
    elif loss3 > max3:
        max3 = loss3
print("DECA:", DECA, DECA / length, max1, min1)
print("EMOCA:", EMOCA, EMOCA / length, max2, min2)
print("DECA_based:", DECA_based, DECA_based / length, max3, min3)
print("EMOCA_based:", EMOCA, EMOCA / length, max4, min4)
# print("HJModel:", HJImage, HJImage / length, max3, min3)
# print("Spectre:", Spectre, Spectre / length, max4, min4)
# print("Ours:", Aours, Aours / length, max5, min5)
# print("Ours_Imag:", Aours_Imag, Aours_Imag / length, max7, min7)
# print("Ours_Seq:", Aours_Seq, Aours_Seq / length, max6, min6)
#
