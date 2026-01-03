import os
import glob
allPath = glob.glob("/home/cine/Documents/ForPaperResult/TestReult/Spectre_AFEW/*/*/lmk/*.npy")
for path in allPath:
    newName = path.replace("frame","0")
    os.rename(path, newName)