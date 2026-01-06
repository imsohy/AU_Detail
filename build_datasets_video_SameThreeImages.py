# 기존 데이터셋 코드에서 이미지 3장을 모두 동일한 이미지로 로드하도록 수정

import os, sys
import random
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
from torch.utils.data import Dataset, ConcatDataset
from glob import glob

import torch.nn.functional as F
def build_train(config, is_train=True):
    data_list = []
    data_list.append(SelfDataset(K=config.K,  image_size=config.image_size , mediapipePath = config.mediapipePath))
    dataset = ConcatDataset(data_list)
    return dataset


class SelfDataset(Dataset):
    def __init__(self, K,  image_size, mediapipePath = 'mediapipe_landmark_embedding.npz'):
        '''
        K must be less than 6
        Modified version: When K=3, all 3 frames use the same image
        '''
        self.mediapipe_idx = \
            np.load(mediapipePath,
                    allow_pickle=True, encoding='latin1')[
                'landmark_indices'].astype(int)
        self.K = K
        # allName = ['FFHQ', 'FFHQ-Aug', 'CelebAHQ', 'CelebAHQ-Aug']
        self.image_size = image_size
        
        ####데이터 소스 위치.###
        self.source1 = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/MEAD_Dataset_25/masks/'
        self.source2 = '/media/cine/First/Aff-wild2/masks/'
        self.MEADAll = glob(os.path.join(self.source1, "*/*/*/*/"))
        random.shuffle(self.MEADAll)

        # self.masksList = self.MEADAll[:3000]+glob(os.path.join(self.source2, "*/*/"))+glob(os.path.join(self.source2, "*/*/"))+glob(os.path.join(self.source2, "*/*/"))+glob(os.path.join(self.source2, "*/*/"))# [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #
        self.masksList = self.MEADAll[:3000] + glob(os.path.join(self.source2, "*/*/")) + glob(
            os.path.join(self.source2, "*/*/")) + glob(os.path.join(self.source2, "*/*/")) + glob(
            os.path.join(self.source2, "*/*/")) + glob(os.path.join(self.source2, "*/*/")) + glob(
            os.path.join(self.source2, "*/*/"))

        random.shuffle(self.masksList)
        self.windowsize = K

    # def shuffle(self):
    #     random.shuffle(self.audioList)

    def __len__(self):
        return len(self.masksList)

    def __getitem__(self, idx):
        # images_224_lists = [];
        images_list = [];
        kpt_list = [];
        dense_kpt_list = [];
        mask_list = [];
        # identity_list= []
        # name = os.path.splitext(os.path.split()[-1])[0]
        # name = os.path.splitext(os.path.split(self.allkptFiles[idx])[-1])[0]
        mask_paths = self.masksList[idx]
        # audio_feature = np.load(audio_path, allow_pickle=True)
        allImagePath = sorted(glob(os.path.join(mask_paths.replace('masks', 'images'),'*.*')))
        if len((allImagePath)) < self.windowsize:
            print("small...", mask_paths)

        if self.windowsize % 2 == 0:
            half1 = half2 = int(self.windowsize / 2)
        else:
            half1 = int(self.windowsize /2)
            half2 = int(self.windowsize /2)+1
        tempId = random.randint(half1, len(allImagePath)-half2)
        # tempId = 24
        imagesName = []
        audio_feature_list = []
        if 'Aff-wild2' in mask_paths:
            mk = 2
        else:
            mk = 1
        
        # MODIFIED: All frames use the same image (tempId)
        for k in range(tempId-half1, tempId+half2):
            # Original: image_path = allImagePath[k]
            # Modified: Use the same image (tempId) for all frames
            image_path = allImagePath[tempId]  # Always use tempId instead of k
            imagesName.append(image_path)
            # if os.path.exists(audio_path.replace('audio', 'images').replace('.npy', '.jpg')):
            #     image_path = audio_path.replace('audio', 'images').replace('.npy', '.jpg')
            # else:
            #     image_path = audio_path.replace('masks', 'images').replace('.npy', '.png')
            # if k == tempId:
            #     identity_path =image_path.replace('images', 'identity').replace('.png', '.npy').replace('.jpg', '.npy')
            #     idName = os.path.splitext(os.path.split(identity_path)[-1])[0]
            # identity_path = image_path.replace('images', 'identity').replace('.png', '.npy').replace('.jpg', '.npy')
            # idName = os.path.splitext(os.path.split(identity_path)[-1])[0]

            kpt_path = image_path.replace('images','kpts').replace('.png', '.npy').replace('.jpg', '.npy')
            kpt_path_mp = image_path.replace('images', 'kpts_dense').replace('.png', '.npy').replace('.jpg', '.npy')
            seg_path = image_path.replace('images', 'masks').replace('.png', '.npy').replace('.jpg', '.npy')
            # audio_path = os.path.split(seg_path.replace('images', 'audio'))[0]

            # dense_lmks =  np.load(kpt_path_mp)[:, :2]
            lmks = np.load(kpt_path)
            dense_lmks = np.load(kpt_path_mp)
            # if not os.path.exists(image_path) or not os.path.exists(kpt_path) or not os.path.exists(kpt_path_mp) or not os.path.exists(seg_path):
            #     print(not os.path.exists(image_path) or not os.path.exists(kpt_path) or not os.path.exists(kpt_path_mp) or not os.path.exists(seg_path))
            if not os.path.getsize(image_path) or not os.path.getsize(kpt_path) or not os.path.getsize(kpt_path_mp) or not os.path.getsize(seg_path):
                print(os.path.getsize(image_path),image_path)
                print(os.path.getsize(kpt_path),kpt_path)
                print(os.path.getsize(kpt_path_mp),kpt_path_mp)
                print(os.path.getsize(seg_path),seg_path)
            image = imread(image_path) / 255.
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            # images_224_lists.append(cv2.resize(image, (224, 224)).transpose(2, 0, 1))
            if image.shape[1] == 224:
                images_list.append(image.transpose(2, 0, 1))
                kpt_list.append(lmks)
                dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
                mask_list.append(mask.transpose(2, 0, 1))
            else:
                kpt_list.append(lmks)
                dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
                mask_list.append(np.resize(mask,(224,224, 3)).transpose(2, 0, 1))
                image = cv2.resize(image,(224,224))
                images_list.append(image.transpose((2, 0, 1)))
            # identity_list.append(np.load(identity_path, allow_pickle=True))
            # audio_feature_list.append(audio_feature[k])

        # images_224_array = torch.from_numpy(np.array(images_224_lists)).type(dtype=torch.float32)
        images_array = torch.from_numpy(np.array(images_list)).type(dtype=torch.float32)
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype=torch.float32)

        dense_kpt_array = torch.from_numpy(np.array(dense_kpt_list)).type(dtype=torch.float32)
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype=torch.float32)
        # identity_array = torch.from_numpy(np.array(identity_list))
        # audio_feature_array = torch.from_numpy(np.array(audio_feature_list))
        # print('id:', idx, '\nfirst name: ',imagesName[0])
        if idx+1==len(self.masksList) or idx == 0:
            random.shuffle(self.MEADAll)
            self.masksList = self.MEADAll[:3000] + glob(os.path.join(self.source2, "*/*/")) + glob(
                os.path.join(self.source2, "*/*/")) + glob(os.path.join(self.source2, "*/*/")) + glob(
                os.path.join(self.source2, "*/*/")) + glob(os.path.join(self.source2, "*/*/")) + glob(
                os.path.join(self.source2, "*/*/"))  # [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #
        # print(imagesName)


        data_dict = {
            'image_224': images_array,
            'imagesName':imagesName,
            # 'image': images_array,
            'landmark': kpt_array,
            'landmark_dense': dense_kpt_array,
            'mask': mask_array,
            # 'au': torch.zeros(0),
            # 'mask': mask_array
        }
        return data_dict


    def load_mask(self, maskpath, h, w):
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)

            # mask = np.zeros_like(vis_parsing_anno)
            mask = np.zeros((h, w, 3))
            # mask = np.zeros((h, w))

            # index = vis_parsing_anno > 0
            mask[vis_parsing_anno > 0] = 1.
            mask[vis_parsing_anno == 4] = 2.
            mask[vis_parsing_anno == 5] = 2.
            mask[vis_parsing_anno == 9] = 2.
            mask[vis_parsing_anno == 7] = 2.
            mask[vis_parsing_anno == 8] = 2.
            mask[vis_parsing_anno == 10] = 0  # hair
            mask[vis_parsing_anno == 11] = 0  # left ear
            mask[vis_parsing_anno == 12] = 0  # right ear
            mask[vis_parsing_anno == 13] = 0  # glasses
            # print('shape...',mask.shape)
        else:
            mask = np.ones((h, w, 3))
            # mask = np.ones((h, w))
        return mask


