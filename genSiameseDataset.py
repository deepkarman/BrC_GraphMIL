import cv2
import numpy as np
import random
import os, sys, glob, pickle, pdb
from xml.dom import minidom
import matplotlib.path as mplPath
import numpy as np
#import openslide
import time
import pdb
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as geo_nn
# from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset, DataLoader
from random import randrange

from PIL import Image
Image.MAX_IMAGE_PIXELS = None



use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class SiamDataset(Dataset):
    def __init__(self, img_file_list, mode='load', affine_param=5, jitter_param=0.4, detailed=False):
        self.img_file_list = img_file_list
        self.detailed = detailed
        self.randomCrop = torchvision.transforms.RandomCrop(224)
        
        self.toTensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        
        self.mode = mode
        if mode=='create':
            self.single_transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor()
            ])
            self.augment = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomAffine(affine_param),
                torchvision.transforms.ColorJitter(
                    brightness=jitter_param,
                    contrast=jitter_param,
                    saturation=jitter_param),
                torchvision.transforms.ToTensor()
            ])

            self.wsi_list = []
            self.wsi_weight = []
            acc = 0
            for img_file in img_file_list:
                wsi = Image.open(img_file).convert('RGB')
                self.wsi_list.append(wsi)
                h,w = wsi.size
                self.wsi_weight.append(h*w)
                # wsi.close()
                acc += 1.*h*w/(224*224)
            self.len_set = int(0.4*acc) # approx same size
            
            
            
        
    def sample(self):
        # img_file = random.choices(self.wsi_list, weights=self.wsi_weight)[0]
        # wsi = Image.open(img_file)#.convert('RGB')
        # x, y = wsi.size
        # matrix=224
        # x1 = randrange(0, x - matrix)
        # y1 = randrange(0, y - matrix)

        # wsi = (wsi.crop((x1, y1, x1 + matrix, y1 + matrix))).convert('RGB')


        wsi = random.choices(self.wsi_list, weights=self.wsi_weight)[0]
        
        img = self.single_transform(wsi)
        # wsi.close()
        
        return img
    
    
    # def sample_detailed(self):
    #     img_file = random.choices(self.wsi_list, weights=self.wsi_weight)[0]
    #     wsi = Image.open(img_file).convert('RGB')
    #     wsi_idx = self.wsi_list.index(wsi)
        
    #     i,j,h,w = self.randomCrop.get_params(wsi, self.randomCrop.size)
        
    #     return wsi_idx, i,j, self.toTensor(F.crop(wsi, i,j,h,w))
               
        
    def __getitem__(self, index):
        if self.mode=='create':
            img1 = self.sample()

            augment = np.random.binomial(1,0.5)

            img2 = self.augment(img1) if augment else self.sample()
        else:
            pkl = open(self.img_file_list[index], "rb")
            img1, img2, augment = pickle.load(pkl)
            pkl.close()
        
        return [img1, img2, augment]
        
        
        
    def __len__(self):
        if self.mode=='create':
            return self.len_set
        
        return len(self.img_file_list)

def create_dataset():
    # img_file_list = glob.glob('/home/karman/DDP/TCGA_expt/TCGA_dataset/*/*.png') 
    # address in himalaya ("/home/Drive3/Karman/TCGA_dataset/TCGA-A1-A0SD-01Z-00-DX1.DB17BFA9-D951-42A8-91D2-F4C2EBC6EB9F/*.png")
    img_file_list = sorted(glob.glob('/home/Drive3/Karman/TCGA_dataset/*/*.png')) 
    # 80% of list
    len_img = len(img_file_list)
    img_file_list = img_file_list[:int(0.1*len_img)] # smaller sample for siam becuase size
    dataset = SiamDataset(img_file_list, mode='create')
    print(len(dataset))

    # out_file_path = '/home/karman/DDP/BrC_GraphMIL/data/tcga_'
    # address in himalaya "data/tcga_"
    out_file_path = 'data/'
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path)

    torch.manual_seed(77077)
    random.seed(71017)

    idx=0
    for data in dataset:
        pickle_out = open(out_file_path+'tcga_'+str(idx),"wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        idx += 1
        if (idx%100==0):
            print("done with ", idx)
        if idx>=len(dataset):
            break


create_dataset()
