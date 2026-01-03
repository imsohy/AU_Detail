import json
import glob
import numpy as np
import os
k = np.load("/home/cine/Documents/ForPaperResult/TestReult/EMOCA/AFEW/01/001/2d_landmark_68/00000.npy",allow_pickle=True)
print(k.shape)
k = np.load("/home/cine/Downloads/AFEW-VA/lmkGT/01/001/00000.npy",allow_pickle=True)
print(k.shape)
alljsonPath = sorted(glob.glob("/home/cine/Downloads/AFEW-VA/images/*/*/*.json"))
for jsonpath in alljsonPath:
    with open(jsonpath,'r') as load_j:
        loadResult = json.load(load_j)
        frames = loadResult["frames"]
        savepath = os.path.split(jsonpath.replace("images", "lmkGT_N"))[0]
        os.makedirs(savepath,exist_ok=True)
        for f in frames:
            if os.path.exists(os.path.join(savepath, f + ".npy")):
                continue
            tformPath = os.path.join(os.path.split(jsonpath.replace("images", "tform"))[0],f+".npy")

            tform = np.load(tformPath)
            # np.save(os.path.join(savepath, f+".npy"),frames[f]['landmarks'])
            lmk = frames[f]['landmarks']
            cropped_kpt = np.dot(tform, np.hstack([lmk, np.ones([68, 1])]).T).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / 224 * 2 - 1

            np.save(os.path.join(savepath, f+".npy"),cropped_kpt)
