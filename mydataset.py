import glob
import random
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean

class myDataset(Dataset):
    def __init__(self, root, mode):
        super(myDataset, self).__init__()
        self.mode = mode
        if self.mode == 'train':
            self.files_S = sorted(glob.glob(os.path.join(root, 'train', 'train_A') + '/*.*'))
            self.files_F = sorted(glob.glob(os.path.join(root, 'train', 'train_C') + '/*.*'))
            self.files_M = sorted(glob.glob(os.path.join(root, 'train', 'train_B') + '/*.*'))
        elif self.mode == 'test':
            self.files_S = sorted(glob.glob(os.path.join(root, 'test', 'test_A') + '/*.*'))
            self.files_F = sorted(glob.glob(os.path.join(root, 'test', 'test_C') + '/*.*'))
            self.files_M = sorted(glob.glob(os.path.join(root, 'test', 'test_B') + '/*.*'))
            self.filesname_S = sorted(os.listdir(os.path.join(root, 'test', 'test_A')))
            self.filesname_F = sorted(os.listdir(os.path.join(root, 'test', 'test_C')))
    def __getitem__(self, index):
        item_S = io.imread(self.files_S[index % len(self.files_F)])
        item_F = io.imread(self.files_F[index % len(self.files_F)])
        item_M = io.imread(self.files_M[index % len(self.files_M)])
        if self.mode == 'test':
            S_name = self.filesname_S[index % len(self.files_F)]
        # F_name = self.filesname_F[index % len(self.files_F)]
            
        item_S = color.rgb2lab(item_S)
        item_S = resize(item_S, (512, 512, 3))
        item_S[:,:,0] = np.asarray(item_S[:,:,0])/50.0-1.0
        item_S[:,:,1:] = 2.0*(np.asarray(item_S[:,:,1:])+128.0)/255.0-1.0
        item_S = torch.from_numpy(item_S.copy()).float()
        item_S = item_S.view(512,512,3)
        item_S = item_S.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_F = color.rgb2lab(item_F)
        item_F = resize(item_F, (512, 512, 3))
        item_F[:,:,0] = np.asarray(item_F[:,:,0])/50.0-1.0
        item_F[:,:,1:] = 2.0*(np.asarray(item_F[:,:,1:])+128.0)/255.0-1.0
        item_F = torch.from_numpy(item_F.copy()).float()
        item_F = item_F.view(512,512,3)
        item_F = item_F.transpose(0, 1).transpose(0, 2).contiguous()
        
        item_M = resize(item_M,(512,512,1))
        item_M[item_M>0] = 1.0
        item_M = np.asarray(item_M)
        item_M = torch.from_numpy(item_M.copy()).float()
        item_M = item_M.view(512,512,1)
        item_M = item_M.transpose(0, 1).transpose(0, 2).contiguous()
        if self.mode == 'test':
            return item_S, item_F, item_M, S_name
        elif self.mode == 'train':
            return item_S, item_F, item_M
    
    def __len__(self):
        return max(len(self.files_S), len(self.files_M))