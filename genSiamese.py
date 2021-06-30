import cv2
import numpy as np
import random
import os, sys, glob, pickle
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

from PIL import Image
Image.MAX_IMAGE_PIXELS = None



use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# MODEL_PATH = '_ckpt_epoch_9.ckpt'
MODEL_PATH = 'tenpercent_resnet18.ckpt'
RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
NUM_CLASSES = 4  # only used if RETURN_PREACTIVATION = False


def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

model = models.__dict__['resnet18'](pretrained=False)

state = torch.load(MODEL_PATH, map_location=device)

state_dict = state['state_dict']
for key in list(state_dict.keys()):
    state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

model = load_model_weights(model, state_dict)

if RETURN_PREACTIVATION:
    model.fc = torch.nn.Sequential()
else:
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.cuda()

class SiameseNetwork(nn.Module):
    def __init__(self, model):
        super(SiameseNetwork, self).__init__()
        self.model = model
    
    def forward_once(self, x):
        out = self.model(x)
        return out
    
    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

class SiamDataset(Dataset):
    def __init__(self, img_file_list, mode='load', affine_param=5, jitter_param=0.4, detailed=False):
        self.img_file_list = img_file_list
        self.detailed = detailed
        self.randomCrop = torchvision.transforms.RandomCrop(224)
        
        self.toTensor = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])
        
        self.mode = mode
        # if mode=='create':
        #     self.single_transform = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomCrop(224),
        #         torchvision.transforms.ToTensor()
        #     ])
        #     self.augment = torchvision.transforms.Compose([
        #         torchvision.transforms.ToPILImage(),
        #         torchvision.transforms.RandomHorizontalFlip(),
        #         torchvision.transforms.RandomAffine(affine_param),
        #         torchvision.transforms.ColorJitter(
        #             brightness=jitter_param,
        #             contrast=jitter_param,
        #             saturation=jitter_param),
        #         torchvision.transforms.ToTensor()
        #     ])

        #     self.wsi_list = []
        #     self.wsi_weight = []
        #     for img_file in img_file_list:
        #         wsi = Image.open(img_file).convert('RGB')
        #         self.wsi_list.append(wsi)
        #         h,w = wsi.size
        #         self.wsi_weight.append(h*w)
            
            
            
        
    def sample(self):
        wsi = random.choices(self.wsi_list, weights=self.wsi_weight)[0]

        img = self.single_transform(wsi)
        
        return img
    
    
    # def sample_detailed(self):
    #     wsi = random.choices(self.wsi_list, weights=self.wsi_weight)[0]
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
            acc = 0
            for wsi in self.wsi_list:
                h,w = wsi.size
                acc += 1.*h*w/(224*224)
            return int(acc)
        
        return len(self.img_file_list)


class ContrastiveLoss(nn.Module):
    # label == 1 means same sample, label == 0 means different samples
    def __init__(self, margin=0., do_average=True):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-6
        self.relu = nn.ReLU()
        self.do_average = do_average
    
    def forward(self, out1, out2, labels):
        dist = (out1 - out2).pow(2).sum(1)
        loss = 0.5*(labels*dist + 
                   (1 + -1.*labels)*self.relu(self.margin - (dist+self.eps).sqrt()).pow(2))
        return loss.mean() if self.do_average else loss.sum()

def train(args, model, device, loss_fn, train_loader, optimizer, num_epochs):  
    
    for epoch in range(num_epochs): 
        epoch_loss = 0.
        for batch_idx, data in enumerate(train_loader):
            x1 = data[0].to(device)
            x2 = data[1].to(device)
            label = data[2].to(device)
            

            optimizer.zero_grad()

            model.train()

            out1, out2 = model(x1, x2)

            loss = loss_fn(out1, out2, label) # calculates the loss
            curr_loss = loss.item()
            loss.backward()
            optimizer.step()
            epoch_loss += curr_loss
            
            del x1
            del x2
            del label
            
            if batch_idx % args['log_interval'] == 0 :
                print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.4f}'.format(
                    epoch, 100. * (batch_idx+1) / len(train_loader), curr_loss))
            if batch_idx+1 == len(train_loader):
                print('\nEpoch: {} Total Loss: {:.4f}\n'.format(epoch, epoch_loss))

args = {}
args['batch_size'] = 150
args['epochs'] = 5
args['seed']=990077
args['lr']=0.01
args['train_ratio']=0.8
args['log_interval']=30

torch.manual_seed(args['seed'])

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

siamNetwork = SiameseNetwork(model)

data_file_list = sorted(glob.glob("data/*"))
dataset = SiamDataset(data_file_list) # default mode is load

train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle = True, **kwargs)

loss_fn = ContrastiveLoss(do_average=False)

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])

train(args, siamNetwork, device, loss_fn, train_loader, optimizer, args['epochs'])

torch.save({
    'epoch': args['epochs'],
    'model_state_dict': siamNetwork.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, 'checkpoints/siamese_5epoch_10percentwsi_lossSum.pth.tar')