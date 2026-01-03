import cv2
import os
from glob import glob
# inputpath = '/media/cine/First/TransformerCode/TestNewIdea/TestReult/pretrainNewIdea_236B/35-30-1920x1080_sequence/18X/result'
# name = 'Actor_03sad_texture' # # angry, calm, disgust, fear, happy, neut, sad, surprise
# inputpath = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI/emoca/Actor_03_sad_output/EMOCA_v2_lr_mse_20/inputs/'
# imagepaths = sorted(glob(os.path.join(inputpath, '*.*')))
# savefolder = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI/emoca/Actor_03_sad_output/EMOCA_v2_lr_mse_20'
name = 'Actor_03sadEMOCA_withVideo' # # angry, calm, disgust, fear, happy, neut, sad, surprise
# inputpath = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI/emoca/spectre_video/EMOCA_v2_lr_mse_20/inputs/'
inputpath = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI/emoca/actor_video/Actor_03_sad/EMOCA_v2_lr_mse_20/inputs'
# inputpath = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI/emoca/Actor_03_happy_output/EMOCA_v2_lr_mse_20/inputs'
imagepaths = sorted(glob(os.path.join(inputpath, '*.*')))
# imagepaths2 = sorted(glob(os.path.join("/media/cine/First/TransformerCode/TestForViT_Video/TestReult/NewTrainViT_Video/Actor_03happy_texture2/result", '*.*')))
imagepaths2 = sorted(glob(os.path.join("/home/cine/Documents/HJCode/AU_sequence/TestReult/OnlyE/pretrain1_/Actor_03sad_texture2/result", '*.*')))
savefolder = '/home/cine/Documents/HJCode/AU_sequence/TestReult/OnlyE/pretrain1_/Actor_03sad_texture2/WithEMOCA'
os.makedirs(savefolder, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(os.path.join(savefolder, name+".mp4"), fourcc, 30, (448 * 6, 448), True)

for i, imagepath in enumerate(imagepaths[1:len(imagepaths)-1]):
    vis_image1 = cv2.imread(imagepaths2[i])
    # print(vis_image1.shape, len(imagepaths), len(imagepaths2))
    # vis_image1 = cv2.resize(vis_image1, (448, 448))
    vis_image1 = vis_image1[:,:448*5,:]
    # vis_image2 = cv2.imread(imagepath.replace('inputs', 'rendered_image'))
    # vis_image2 = cv2.resize(vis_image2, (448, 448))
    vis_image3 = cv2.imread(imagepath.replace('inputs', 'geometry_coarse'))
    vis_image3 = cv2.resize(vis_image3, (448, 448))
    # print(vis_image2.shape, vis_image1.shape)
    # vis_image = cv2.hconcat([vis_image1,vis_image2, vis_image3])
    vis_image = cv2.hconcat([vis_image1,vis_image3])
    out.write(vis_image)
out.release()
