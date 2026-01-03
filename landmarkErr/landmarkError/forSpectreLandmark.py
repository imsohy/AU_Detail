import numpy as np
from glob import glob
import os
spectreDir = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*/*/result"
spectreDir_lmk = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*/*/lmk"
allNpy = sorted(glob(spectreDir_lmk+"/*.npy"))
allImage = sorted(glob(spectreDir+"/*.jpg"))
for p in allNpy:
    landmarks = np.load(p)
    if len(landmarks[2:-2]) == len(glob(os.path.split(p)[0]+"/*.jpg")):
        for count, lmk in enumerate(landmarks[2:-2]):
            np.save(os.path.join(os.path.split(p)[0],f'frame{count:04d}.npy'),lmk )
    else:

        print(len(landmarks[2:-2]))
        print(len(glob(os.path.split(p)[0] + "/*.jpg")))
        print(p)
    print("_")
    # os.rename(p,p.replace(".npy","_.npy"))

# landmarkpath_spectre = "/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/01/001/result/frame0000.npy"
#
# np.load(landmarkpath_spectre)