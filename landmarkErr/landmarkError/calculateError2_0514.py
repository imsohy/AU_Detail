import os.path

from tensor_cropper import transform_points, batch_kp_2d_l1_loss
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm
landmarkDir = sorted(glob("/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/*/*/2d_landmark_68/*.npy"))
landmarkDir_EMOCA = "/home/cine/Documents/ForPaperResult/TestReult/EMOCA/AFEW/*"
landmarkDir_spectre = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*"
# landmarkDir_HJ = "/home/cine/Documents/ForPaperResult/TestReult/HJResult/AFEW/01/001/2d_landmark_68/00000.npy"
landmarkDir_Seq = "/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/sequence_pretrain6/AFWE_VA/*"
landmarkDir_Imag = "/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/OnlyE0510/AFWE_VA/*"
# landmarkDir_Imag = "/home/cine/Documents/ForPaperResult/TestReult_New0420/AULoss/Image_pretrain4/AFWE_VA/*"
# /home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/01/001/lmk/frame0001.npy
# landmarkpathGT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy"
landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"
# tformpath = "/home/cine/Downloads/AFEW-VA/tform/01/001/00001.npy"
Aours = 0.
Aours_Imag = 0.
Aours_Seq = 0.
DECA = 0.
EMOCA = 0.
Spectre = 0.
HJImage = 0.
length = len(landmarkDir)
max1 = 0
min1 = 1000
max2 = 0
min2 = 1000
max3 = 0
min3 = 1000
max4 = 0
min4 = 1000
max5 = 0
min5 = 1000
max6 = 0
min6 = 1000
max7 = 0
min7 = 1000
for landmarkpath in tqdm(landmarkDir):
    if "DECA" in landmarkpath:
        length-=1
        continue
    landmarks2d = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))
    name = os.path.splitext(os.path.split(landmarkpath)[-1])[0]
    #/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/01/001/2d_landmark_68/00001_DECA.npy
    landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace(".npy","_DECA.npy"), allow_pickle=True))
    landmarkpath_Seq = landmarkDir_Seq.replace("*",landmarkpath.split("pretrain5X_25/")[-1])
    landmarkpath_Imag = landmarkDir_Imag.replace("*",landmarkpath.split("pretrain5X_25/")[-1])
    landmarkpath_EMOCA = landmarkDir_EMOCA.replace("*",landmarkpath.split("pretrain5X_25/")[-1])
    landmarkpath_Spectre = landmarkDir_spectre.replace("*",landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/","lmk/"))
    landmarkpathGT = landmarkDir_GT.replace("*", landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/",""))
    if not os.path.exists(landmarkpath_Seq) or not os.path.exists(landmarkpath_Imag):
        continue
    landmarks2d_Seq = torch.from_numpy(np.load(landmarkpath_Seq, allow_pickle=True))
    landmarks2d_Imag = torch.from_numpy(np.load(landmarkpath_Imag, allow_pickle=True))
    landmarks2d_EMOCA = torch.from_numpy(np.load(landmarkpath_EMOCA, allow_pickle=True))
    landmarks2d_Spectre = torch.from_numpy(np.load(landmarkpath_Spectre, allow_pickle=True))
    landmarks2dGT = torch.from_numpy(np.load(landmarkpathGT, allow_pickle=True) )

    loss1 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_DECA[17:])
    loss2 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_EMOCA[0][17:])
    loss6 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_Seq[17:])
    loss7 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_Imag[17:])
    loss4 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d_Spectre[17:])
    loss5 = batch_kp_2d_l1_loss(landmarks2dGT[17:], landmarks2d[17:])
    DECA += loss1
    EMOCA += loss2
    # HJImage += loss3
    Spectre += loss4
    Aours += loss5
    Aours_Seq += loss6
    Aours_Imag += loss7
    if loss1 < min1:
        min1 = loss1
    elif loss1 > max1:
        max1 = loss1
    if loss2 < min2:
        min2 = loss2
    elif loss2 > max2:
        max2 = loss2
    if loss6 < min6:
        min6 = loss6
    elif loss6 > max6:
        max6 = loss6
    if loss7 < min7:
        min7 = loss7
    elif loss7 > max7:
        max7 = loss7
    if loss4 < min4:
        min4 = loss4
    elif loss4 > max4:
        max4 = loss4
    if loss5 < min5:
        min5 = loss5
    elif loss5 > max5:
        max5 = loss5
print("DECA:", DECA, DECA / length, max1, min1)
print("EMOCA:", EMOCA, EMOCA / length, max2, min2)
# print("HJModel:", HJImage, HJImage / length, max3, min3)
print("Spectre:", Spectre, Spectre / length, max4, min4)
print("Ours:", Aours, Aours / length, max5, min5)
print("Ours_Imag:", Aours_Imag, Aours_Imag / length, max7, min7)
print("Ours_Seq:", Aours_Seq, Aours_Seq / length, max6, min6)

