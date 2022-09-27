import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as F_t

from PIL import Image
import random
import os
import numpy as np
import math 
import h5py

from starter_code.utils import load_case,load_volume
from starter_code.visualize import visualize
from starter_code.evaluation import evaluate
# from elasticdeform import deform_random_grid
import cv2


class Kits_Dataset(Dataset):
    def __init__(self,directory,resize=False,random_crop=True,crop_r=2,mode = 'train'):
        super(Kits_Dataset,self).__init__()
        self.directory = os.path.join(directory,mode)
        self.file_list = os.listdir(self.directory)
        self.file_list.sort()
        self.resize = resize
        self.random_crop = random_crop
        self.mode = mode
        self.crop_r = crop_r
        self.mean = -521.90679
        self.std = 533.8294680
    def __len__(self):
        return len(self.file_list)
    def transforms(self,vol,seg):
        vol = torch.from_numpy(vol)
        seg = torch.from_numpy(seg)
        if (self.random_crop) and (self.mode == 'train'):
            crop_size = int(len(vol)/self.crop_r)
            i,j,h,w = torchvision.transforms.RandomCrop(crop_size).get_params(vol,(crop_size,crop_size))
            vol = vol[i:i+h,j:j+w]
            seg = seg[i:i+h,j:j+w]
        if self.mode == 'train':
            if random.random() < 0.5 :
                vol = torch.fliplr(vol)
                seg = torch.fliplr(seg)
            if random.random() < 0.5 :
                vol = torch.flipud(vol)
                seg = torch.flipud(seg)
            # [vol,seg] = deform_random_grid([vol,seg],sigma=15,mode='nearest') # Code for elastic deformation
            # seg = np.round(seg)
            # seg[seg<0] = 0
            # seg[seg>2] = 2
        return vol,seg

    def __getitem__(self,idx):
        imgs = np.load(os.path.join(self.directory,self.file_list[idx]))
        seg = imgs[0]
        vol = imgs[1]
        vol = (vol-self.mean)/self.std
        vol,seg = self.transforms(vol,seg)
        vol = vol.unsqueeze(0)
        seg = seg.unsqueeze(0)
        return vol,seg
class UnetSkipConnetion(nn.Module):
    def __init__(self,input_dim,output_dim,submodule=None):
        super(UnetSkipConnetion,self).__init__()
        self.submodule = submodule
        self.enc = nn.Sequential(
            nn.Conv2d(input_dim,output_dim,3,1,1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(output_dim,output_dim,3,1,1),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2,True)
        )
        self.dec = nn.Sequential(
            nn.Conv2d(output_dim * 2,output_dim,3,1,1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True),
            nn.Conv2d(output_dim,output_dim,3,1,1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True)
        )
        self.pre_up_conv = nn.Sequential(
            nn.ConvTranspose2d(output_dim*2,output_dim,4,2,1)
        )
    def forward(self,x):
        enc = self.enc(x)
        if self.submodule == None :
            return enc
        else :
            sub_in = F.max_pool2d(enc,2)
            sub = self.submodule(sub_in)
            sub = self.pre_up_conv(sub)
            dec_in = torch.cat([enc,sub],dim=1)
            dec = self.dec(dec_in)
            return dec
class U_Net(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers, residual_path):
        super(U_Net,self).__init__()
        self.residual_path = residual_path
        curr_dim = 16
        self.in_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim,curr_dim,7,1,0),
            nn.LeakyReLU(0.2,True)
        )
        curr_dim = curr_dim * 2**(num_layers-1)
        for i in range(num_layers-1,-1,-1):
            if i == num_layers-1 :
                setattr(self,'unet_'+str(i),UnetSkipConnetion(curr_dim,curr_dim*2,submodule=None))
            else :
                setattr(self,'unet_'+str(i),UnetSkipConnetion(curr_dim,curr_dim*2,submodule=getattr(self,'unet_'+str(i+1))))
            curr_dim //= 2
        curr_dim = 32
        self.out_conv = nn.Conv2d(32,input_dim,3,1,1)
        self.out_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim,output_dim,7,1,0)
        )
    def forward(self,x):
        x = x.float()
        x = self.in_layer(x)
        x = self.unet_0(x)
        x = self.out_layer(x)
        out = x
        # if self.residual_path == True :
        #     out = out + x
        return out