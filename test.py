from util.util import  psnr, rmse
from skimage.metrics import structural_similarity as ssim
import glob
from skimage import io, color
import numpy as np
from numpy import *

f_impath = "/home/LeiFeng/zwl/shadow_remove/datasets/istdplus/testF"
s_impath = "/home/LeiFeng/zwl/shadow_remove/datasets/istdplus/testS"



spath = []
fpath = []

for s_img in glob.glob(s_impath+"/*"):
    spath.append(s_img)
for f_img in glob.glob(f_impath+"/*"):
    fpath.append(f_img)


x = []
y = []
z = []

for i in range(len(spath)):

    # import pdb;pdb.set_trace()
    S_img = io.imread(spath[i])
    F_img = io.imread(fpath[i])
    # import pdb;pdb.set_trace()
    S_img = np.asarray(S_img)
    F_img = np.asarray(F_img)
    # import pdb;pdb.set_trace()
    x.append(ssim(S_img, F_img, channel_axis=-1))
    y.append(psnr(S_img, F_img))
    z.append(rmse(S_img, F_img))

print("ssim:", mean(x), "psnr:", mean(y), "rmse:", mean(z))