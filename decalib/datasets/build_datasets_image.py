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
        '''
        self.mediapipe_idx = \
            np.load(mediapipePath,
                    allow_pickle=True, encoding='latin1')[
                'landmark_indices'].astype(int)
        self.K = K
        # allName = ['FFHQ', 'FFHQ-Aug', 'CelebAHQ', 'CelebAHQ-Aug']
        self.image_size = image_size
        #
        # self.source1 = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/MEAD_Dataset_25/masks/'
        # self.source2 = '/media/cine/First/Aff-wild2/masks/'
        # self.masksList = (glob(os.path.join(self.source1, "*/*/*/*/")))+(glob(os.path.join(self.source2, "*/*/")))# [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #

        # self.source1 = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI_Dataset/CelebAHQ/masks/'
        # self.source2 = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI_Dataset/FFHQ/masks/'
        self.source1 = '/media/cine/First/LGAI_Dataset/CelebAHQ/masks/'
        self.source2 = '/media/cine/First/LGAI_Dataset/FFHQ/masks/'
        self.source3 = '/media/cine/First/LGAI_Dataset/FFHQ_Aug/masks/'
        self.source4 = '/media/cine/First/LGAI_Dataset/CelebAHQ_Aug/masks/'
        self.source5 = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/LGAI_Dataset/BUPT/masks/'
        self.r3 = glob(os.path.join(self.source3, "*.*"))
        self.r4 = glob(os.path.join(self.source4, "*.*"))
        random.shuffle(self.r3)
        random.shuffle(self.r4)
        # self.MEADAll = glob(os.path.join(self.source1, "*/*/level_3/*/*.*"))
        # random.shuffle(self.MEADAll)
        self.masksList = ((glob(os.path.join(self.source1, "*.*"))) + (glob(os.path.join(self.source2, "*.*"))) +glob(os.path.join(self.source5,'*.*'))
                          +(self.r3[:10000]) + (self.r4[:10000]))#+(self.MEADAll[:50000]))# [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #

        # self.audioList = sorted(glob(os.path.join(self.source, "*/*/*/*.*")))[7810+13840+57:7810+13840+57+90]#[736+782+615:]# [267+219+58:267+219+58+90] #
        # self.audioList = sorted(glob(os.path.join(self.source, "*/*/*/*.*")))[16994:16994+2]# [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #
        # self.audioList = (glob(os.path.join(self.source2, "*/*/*.*")))# [26400:26400+40] #[736+782+615:]# [267+219+58:267+219+58+90] #
        # self.audioList = sorted(glob(os.path.join(self.source, "*/*/*/*.*")))[29117:29117+1] #[736+782+615:]# [267+219+58:267+219+58+90] #
        # self.source = '/media/cine/de6afd1d-c444-4d43-a787-079519ace719/MEAD_Dataset_25/audio_feature/M003/'
        # self.audioList = glob(os.path.join(self.source, "*/*/*.*"))
        # self.allmasksFolder = (
        #             glob(self.source[0] + '/*.npy') + glob(self.source[1] + '/*.npy')
        #             # + glob(self.source[2] + '/*.npy') + glob(self.source[3] + '/*.npy')
        #             + glob(self.source[2] + '*/*/*.npy')
        #             + glob(self.source[3] + '*/*/*/*/*.npy')
        # )

        random.shuffle(self.masksList)
        self.windowsize = 16

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
        # mask_paths = self.masksList[idx]
        # audio_feature = np.load(audio_path, allow_pickle=True)
        imagesName = []
        mask_paths = self.masksList[idx]
        if os.path.exists(mask_paths.replace('masks', 'images').replace('.npy', '.png')):
            image_path = mask_paths.replace('masks', 'images').replace('.npy', '.png')
        elif os.path.exists(mask_paths.replace('masks', 'images').replace('.npy', '.jpg')):
            image_path = mask_paths.replace('masks', 'images').replace('.npy', '.jpg')
        imagesName.append(image_path)

        kpt_path = image_path.replace('images','kpts').replace('.png', '.npy').replace('.jpg', '.npy')
        kpt_path_mp = image_path.replace('images', 'kpts_dense').replace('.png', '.npy').replace('.jpg', '.npy')
        seg_path = image_path.replace('images', 'masks').replace('.png', '.npy').replace('.jpg', '.npy')
        # identity_path = seg_path.replace('masks', 'identity')
        # audio_path = os.path.split(seg_path.replace('images', 'audio'))[0]

        # dense_lmks =  np.load(kpt_path_mp)[:, :2]
        lmks = np.load(kpt_path)
        dense_lmks = np.load(kpt_path_mp)
        image = imread(image_path) / 255.
        mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

        # images_224_lists.append(cv2.resize(image, (224, 224)).transpose(2, 0, 1))
        if image.shape[1] == 224:
            images_list.append(image.transpose(2, 0, 1))
            kpt_list.append(lmks)
            dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
            mask_list.append(mask)
        else:
            kpt_list.append(lmks)
            dense_kpt_list.append(dense_lmks[self.mediapipe_idx, :])
            mask_list.append(np.resize(mask,(224,224,3)))
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
        if idx==0 or idx+1==len(self.masksList):
            print('shuffle', self.masksList[0])
            random.shuffle(self.r3)
            random.shuffle(self.r4)
            # random.shuffle(self.MEADAll)
            self.masksList = (
                        (glob(os.path.join(self.source1, "*.*"))) + (glob(os.path.join(self.source2, "*.*"))) + glob(
                    os.path.join(self.source5, '*.*')) +(self.r3[:10000]) + (self.r4[:10000]))#+(self.MEADAll[:50000]))

            random.shuffle(self.masksList)
        data_dict = {
            # 'image_224': images_224_array,
            'imagesName':imagesName,
            'image_224': images_array,
            'landmark': kpt_array,
            'landmark_dense': dense_kpt_array,
            'mask': mask_array,
            # 'audio_feature':audio_feature_array,
            # 'identity':torch.tensor(np.load(identity_path, allow_pickle=True))
            # 'identity':identity_array[0]
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


