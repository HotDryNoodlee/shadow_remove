import os.path
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import util.util as util
import glob
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import cv2

class lightDataset(BaseDataset):
    
    def __init__(self, opt):

        self.opt = opt
        if self.opt.phase == "train":
            if self.opt.use_mask == True:
                self.dir_S = os.path.join(self.opt.dataroot, self.opt.phase + 'S')  # create a path '/path/to/data/trainA'
                self.dir_F = os.path.join(self.opt.dataroot, self.opt.phase + 'F')  # create a path '/path/to/data/trainB'
                self.dir_M = os.path.join(self.opt.dataroot, self.opt.phase + 'M')  # create a path '/path/to/data/mask'

                self.S_paths = sorted(make_dataset(self.dir_S, self.opt.max_dataset_size))   # load images from '/path/to/data/trainA'
                self.F_paths = sorted(make_dataset(self.dir_F, self.opt.max_dataset_size))    # load images from '/path/to/data/trainB'
                self.M_paths = sorted(make_dataset(self.dir_M, self.opt.max_dataset_size))    # load images from '/path/to/data/mask'

                self.S_size = len(self.S_paths)  # get the size of dataset A
                self.F_size = len(self.F_paths)  # get the size of dataset B
                self.M_size = len(self.M_paths)  # get the size of dataset M
                
            else:
                self.dir_S = os.path.join(self.opt.dataroot, self.opt.phase + 'S')  # create a path '/path/to/data/trainA'
                self.dir_F = os.path.join(self.opt.dataroot, self.opt.phase + 'F')  # create a path '/path/to/data/trainB'

                self.S_paths = sorted(make_dataset(self.dir_S, self.opt.max_dataset_size))   # load images from '/path/to/data/trainA'
                self.F_paths = sorted(make_dataset(self.dir_F, self.opt.max_dataset_size))    # load images from '/path/to/data/trainB'

                self.S_size = len(self.S_paths)  # get the size of dataset A
                self.F_size = len(self.F_paths)  # get the size of dataset B
        
        if self.opt.phase == "test":
            if self.opt.use_mask:
                self.dir_S = os.path.join(self.opt.dataroot, self.opt.phase + 'S')
                self.dir_F = os.path.join(self.opt.dataroot, self.opt.phase + 'F')
                self.dir_M = os.path.join(self.opt.dataroot, self.opt.phase + 'M')

                self.S_paths = sorted(make_dataset(self.dir_S, self.opt.max_dataset_size))   # load images from '/path/to/data/testA'
                self.F_paths = sorted(make_dataset(self.dir_F, self.opt.max_dataset_size))    # load images from '/path/to/data/testB'
                self.M_paths = sorted(make_dataset(self.dir_M, self.opt.max_dataset_size))

                self.S_size = len(self.S_paths)  # get the size of dataset A
                self.F_size = len(self.F_paths)  # get the size of dataset B
                self.M_size = len(self.M_paths)
            
            else:
                self.dir_S = os.path.join(self.opt.dataroot, self.opt.phase + 'S')
                self.dir_F = os.path.join(self.opt.dataroot, self.opt.phase + 'F')

                self.S_paths = sorted(make_dataset(self.dir_S, self.opt.max_dataset_size))   # load images from '/path/to/data/testA'
                self.F_paths = sorted(make_dataset(self.dir_F, self.opt.max_dataset_size))    # load images from '/path/to/data/testB'

                self.S_size = len(self.S_paths)  # get the size of dataset A
                self.F_size = len(self.F_paths)  # get the size of dataset B


    def __getitem__(self, index):

        if self.opt.use_mask:
            S_path = self.S_paths[index % self.S_size]  # make sure index is within then range
            F_path = self.F_paths[index % self.F_size]
            M_path = self.M_paths[index % self.M_size]
            
            S_img = cv2.imread(S_path)
            F_img = cv2.imread(F_path)
            M_img = cv2.imread(M_path)

            S_img = cv2.cvtColor(S_img, cv2.COLOR_BGR2RGB)
            S_img = cv2.cvtColor(S_img, cv2.COLOR_RGB2LAB)
            F_img = cv2.cvtColor(F_img, cv2.COLOR_BGR2RGB)
            F_img = cv2.cvtColor(F_img, cv2.COLOR_RGB2LAB)
            
            # import pdb; pdb.set_trace()
            S_img = resize(S_img, (self.opt.loadsize, self.opt.loadsize, 3))
            F_img = resize(F_img, (self.opt.loadsize, self.opt.loadsize, 3))
            M_img = resize(M_img, (self.opt.loadsize, self.opt.loadsize, 1))

            M_img[M_img>0] = 1.0

            S_img[:,:,0] = np.asarray(S_img[:,:,0])/50.0-1.0
            F_img[:,:,0] = np.asarray(F_img[:,:,0])/50.0-1.0
            S_img[:,:,1:] = 2.0*(np.asarray(S_img[:,:,1:])+128.0)/255.0-1.0
            F_img[:,:,1:] = 2.0*(np.asarray(F_img[:,:,1:])+128.0)/255.0-1.0

            S_img = torch.from_numpy(S_img.copy()).float()
            F_img = torch.from_numpy(F_img.copy()).float()
            M_img = torch.from_numpy(M_img.copy()).float()

            S_img = S_img.view(self.opt.loadsize,self.opt.loadsize,3)
            F_img = F_img.view(self.opt.loadsize,self.opt.loadsize,3)
            M_img = M_img.view(self.opt.loadsize, self.opt.loadsize,1)

            S_img = S_img.transpose(0, 1).transpose(0, 2).contiguous()
            F_img = F_img.transpose(0, 1).transpose(0, 2).contiguous()
            M_img = M_img.transpose(0, 1).transpose(0, 2).contiguous()

            S = S_img
            F = F_img
            M = M_img

            return {'S': S, 'F': F, 'M':M,  'S_paths': S_path, 'F_paths': F_path, 'M_paths': M_path}

        else:
            S_path = self.S_paths[index % self.S_size]  # make sure index is within then range
            F_path = self.F_paths[index % self.F_size]
            
            S_img = cv2.imread(S_path)
            F_img = cv2.imread(F_path)
            # import pdb;pdb.set_trace()
            S_img = cv2.resize(S_img, (self.opt.loadsize, self.opt.loadsize))
            F_img = cv2.resize(F_img, (self.opt.loadsize, self.opt.loadsize))

            S_img = np.asarray(S_img)/255
            F_img = np.asarray(F_img)/255

            S_img = torch.from_numpy(S_img.copy()).float()
            F_img = torch.from_numpy(F_img.copy()).float()

            S_img = S_img.transpose(0, 1).transpose(0, 2).contiguous()
            F_img = F_img.transpose(0, 1).transpose(0, 2).contiguous()

            S = S_img
            F = F_img

            return {'S': S, 'F': F, 'S_paths': S_path, 'F_paths': F_path}

    def __len__(self):

        return max(self.S_size, self.F_size)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    for path in sorted(glob.glob(dir+"/*")):
        images.append(path)
    return images[:min(max_dataset_size, len(images))]