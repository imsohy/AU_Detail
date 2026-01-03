import os.path
import numpy as np
import cv2
import torch
from glob import glob
from tqdm import tqdm

def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 2
    kp_pred: N x K x 2
    """
    kp_gt = real_2d_kp.view(-1,3)
    # print('true_content1_1_1', real_2d_kp[0][1][1])
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    # print('kpt_content1_1_1', kp_pred[0][1][1])
    vis = kp_gt[:,2]
    k = torch.sum(vis) * 2.0 + 1e-8

    dif_abs = torch.abs(kp_gt[:,:2] - kp_pred).sum(1)
    # return torch.matmul(dif_abs, vis) * 1.0 / k
    return dif_abs
landmarkDir = sorted(glob("/home/cine/Documents/ForPaperResult/TestReult/AFWE_VA/pretrain5X_25/*/*/2d_landmark_68/*.npy"))
landmarkDir_EMOCA = "/home/cine/Documents/ForPaperResult/TestReult/EMOCA/AFEW/*"
landmarkDir_spectre = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*"
# landmarkDir_HJ = "/home/cine/Documents/ForPaperResult/TestReult/HJResult/AFEW/01/001/2d_landmark_68/00000.npy"
landmarkDir_HJ = "/home/cine/Documents/ForPaperResult/TestReult/HJResult/AFEW/*"
# /home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/01/001/lmk/frame0001.npy
# landmarkpathGT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/01/001/00001.npy"
landmarkDir_GT = "/home/cine/Downloads/AFEW-VA/lmkGT_N/*"
# tformpath = "/home/cine/Downloads/AFEW-VA/tform/01/001/00001.npy"
length = len(landmarkDir)
heatmapSize = 56
for landmarkpath in tqdm(landmarkDir):
    if "DECA" in landmarkpath:
        length-=1
        continue
    landmarks2d = torch.from_numpy(np.load(landmarkpath, allow_pickle=True))
    # image = cv2.imread(landmarkpath.replace("2d_landmark_68", "result",).replace(".npy",".jpg"))[:,:448]
    heatmap = np.zeros([heatmapSize,heatmapSize,5])
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
    ourL =batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d)
    DECAL =batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_DECA)
    HJImageL =batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_HJ)
    EMOCAL =batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_EMOCA)
    SpectreL =batch_kp_2d_l1_loss(landmarks2dGT[None,...], landmarks2d_Spectre)
    tBlandmarks2dGT = (landmarks2dGT+1)/2 *heatmapSize
    for i, lmk in enumerate(tBlandmarks2dGT):
        x = min(int(lmk[0]),heatmapSize-1)
        y = min(int(lmk[1]), heatmapSize-1)
        heatmap[y, x,0] = ourL[i]
        heatmap[y, x,1] = DECAL[i]
        heatmap[y, x,2] = HJImageL[i]
        heatmap[y, x,3] = EMOCAL[i]
        heatmap[y, x,4] = SpectreL[i]
        # heatmap[y, x,0] = ourL[i]
    m = ["our","DECA","HJ","EMOCA","Spectre"]
    heatmapA = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    for i in range(5):
        heatA = cv2.applyColorMap(heatmapA[:,:,i], cv2.COLORMAP_HOT)
        # for i in range(448):
        #     for j in range(448):
        #         # print(heatA[i][j])
        #         if heatA[i][j][0] == 128 and heatA[i][j][1]==0 and heatA[i][j][2]==0:
        #             heatA[i][j] = [255,255,255]
        cv2.imwrite("result_"+m[i]+".jpg", heatA)



