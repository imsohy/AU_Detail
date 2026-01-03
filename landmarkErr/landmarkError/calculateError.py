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
landmarkDir_HJ = "/home/cine/Documents/ForPaperResult/TestReult/HJResult/AFEW/*"
# /home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/01/001/lmk/frame0001.npy
# landmarkpathGT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy"
landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"
# tformpath = "/home/cine/Downloads/AFEW-VA/tform/01/001/00001.npy"
Aours = 0.
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

for landmarkpath in tqdm(landmarkDir):
    if "DECA" in landmarkpath:
        length-=1
        continue
    landmarks2d = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))
    name = os.path.splitext(os.path.split(landmarkpath)[-1])[0]
    #/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/01/001/2d_landmark_68/00001_DECA.npy
    landmarks2d_DECA = torch.from_numpy(np.load(landmarkpath.replace(".npy","_DECA.npy"), allow_pickle=True))
    landmarkpath_HJ = landmarkDir_HJ.replace("*",landmarkpath.split("pretrain5X_25/")[-1])
    landmarkpath_EMOCA = landmarkDir_EMOCA.replace("*",landmarkpath.split("pretrain5X_25/")[-1])
    landmarkpath_Spectre = landmarkDir_spectre.replace("*",landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/","lmk/"))
    landmarkpathGT = landmarkDir_GT.replace("*", landmarkpath.split("pretrain5X_25/")[-1].replace("2d_landmark_68/",""))
    landmarks2d_HJ = torch.from_numpy(np.load(landmarkpath_HJ, allow_pickle=True))
    landmarks2d_EMOCA = torch.from_numpy(np.load(landmarkpath_EMOCA, allow_pickle=True))
    landmarks2d_Spectre = torch.from_numpy(np.load(landmarkpath_Spectre, allow_pickle=True))
    landmarks2dGT = torch.from_numpy(np.load(landmarkpathGT, allow_pickle=True) )

    loss1 = batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_DECA)
    loss2 = batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_EMOCA)
    loss3 = batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_HJ)
    loss4 = batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_Spectre)
    loss5 = batch_kp_2d_l1_loss(landmarks2dGT[None, ...], landmarks2d)

    DECA +=loss1
    EMOCA += loss2
    HJImage +=loss3
    Spectre += loss4
    Aours += loss5
    if loss1 < min1:
        min1 = loss1
    elif loss1 > max1:
        max1 = loss1
    if loss2 < min2:
        min2 = loss2
    elif loss2 > max2:
        max2 = loss2
    if loss3 < min3:
        min3 = loss3
    elif loss3 > max3:
        max3 = loss3
    if loss4 < min4:
        min4 = loss4
    elif loss4 > max4:
        max4 = loss4
    if loss5 < min5:
        min5 = loss5
    elif loss5 > max5:
        max5 = loss5
print("DECA:",DECA, DECA/length, max1, min1)
print("EMOCA:",EMOCA, EMOCA/length, max2, min2)
print("HJModel:",HJImage, HJImage/length, max3, min3)
print("Spectre:",Spectre, Spectre/length, max4, min4)
print("Ours:",Aours, Aours/length, max5, min5)

