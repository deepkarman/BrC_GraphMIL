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
import pandas as pd
import torch_geometric.transforms as T
import torch_geometric.nn as geo_nn
# from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset, DataLoader

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SAMPLES_PER_IMAGE = 2000

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

MODEL_PATH = 'tenpercent_resnet18.ckpt'
RETURN_PREACTIVATION = True  # return features from the model, if false return classification logits
NUM_CLASSES = 4  # only used if RETURN_PREACTIVATION = False

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

def generate_graph_label(label):
    # TODO - rewrite this 
    n_stage = label['N stage'].iloc[0]
    m_stage = label['M stage'].iloc[0]

    if n_stage[0:2] == 'NX' or m_stage[0:2] == 'MX':
        return -1 

    if n_stage[0:2] == 'N0' and m_stage[0:2] == 'M0':
        return 1

    if (n_stage[0:2] == 'N1' or n_stage[0:2] == 'N2' or n_stage[0:2] == 'N3') and m_stage[0:2] == 'M0':
        return 0

    if n_stage[0:2] == 'N3' and m_stage[0:2] == 'M1':
        return 0

    raise Exception("Unknown Label")


# TODO
class SiamDataset(Dataset):
    def __init__(self, img_file, mode='load', affine_param=5, jitter_param=0.4, detailed=False):
        self.img_file = img_file
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

            self.wsi = Image.open(img_file).convert('RGB')

                         
        
    def sample(self):
        # wsi = random.choices(self.wsi_list, weights=self.wsi_weight)[0]
        
        img = self.single_transform(self.wsi)
        
        return img
               
        
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
        global SAMPLES_PER_IMAGE
        return SAMPLES_PER_IMAGE # we want 2000 per image
        # if self.mode=='create':
        #     return self.len_set
        
        # return len(self.img_file_list)




siamNetwork = SiameseNetwork(model)

checkpoint = torch.load('checkpoints/siamese_5epoch_10percentwsi_lossSum.pth.tar')
siamNetwork.load_state_dict(checkpoint['model_state_dict'])


# img_file_list = glob.glob('/home/karman/DDP/TCGA_expt/TCGA_dataset/*/*.png')

path = '/home/Drive3/Karman/TCGA_dataset/'
save_path = '/home/Drive3/Karman/BrC_GraphMIL/data_graphs/'
slides = sorted(glob.glob(path+'*'))

label_sheet_name = '/home/Drive3/Karman/TCGA_master_allcases_2019.06.13 NK 12-09-19 Test-train cases marked_NC_added_mag_28-07-20.xlsx'
xl_sheet = pd.read_excel(label_sheet_name, sheet_name='All', header = 1, nrows = 1006, usecols ='D,Y:Z')
count = 0
slide_idx = -1

for slide in slides:
    images = sorted(glob.glob(slide+'/*.png'))

    slide_label_name = slide.rsplit('/',1)[-1][0:12]
    slide_label_df = xl_sheet[xl_sheet['ID'] == slide_label_name]
    # pdb.set_trace()
    # set_labels.append([slide_label_df['N stage'].iloc[0], slide_label_df['M stage'].iloc[0]])
    slide_label = generate_graph_label(slide_label_df)
    if slide_label == -1:
        continue
    slide_idx += 1

    for image in images:
        data_curr = [[], [], []]      
        file_name_curr = image.rsplit('/',1)[1] 

        done_this = True

        num_graphs = 7 # from reference

        for idx in range(num_graphs):
            # save graphs
            save_path_curr = save_path + image.rsplit('/',2)[1]+'/'+'idx'+str(idx)+'__'+file_name_curr.replace('.png','.pickle')

            if os.path.exists(save_path_curr):
                count = count + 1
                print("Already exists: ", count, file_name_curr)
            else:
                done_this = False
        if done_this:
            continue
      
        dataset = SiamDataset(image, mode='create') # should be one image

        # there is atleast one roi that is smaller than 100x100 pixels
        if min(dataset.wsi.size) < 224:
            continue

        num_samples = SAMPLES_PER_IMAGE
        
        sample_size = 512

        vertices = torch.Tensor(num_samples, sample_size)
        # print(x_cluster)
        # torch.cat(data_arr, dim=0, out=x_cluster)
        # #torch.cat(data_arr, dim=0)
        # print(x_cluster.shape)


        siamNetwork.eval()

        with torch.no_grad():
            for idx in range(num_samples):
                vertices[idx,:] = siamNetwork.forward_once(dataset.sample().unsqueeze(dim=0).to(device))

        num_graphs = 7 # from reference
        dist_thresh = 0.0037 # for about 15 edges per node

        vertices = vertices.chunk(num_graphs) 

        edges = []
        for vertex in vertices:
            dist = torch.cdist(vertex, vertex, 2)
            i, j = np.where((dist<dist_thresh))
            edge = np.array([i,j])
            edges.append(edge)

        graphs = []
        for idx in range(num_graphs):
            # save graphs
            save_path_curr = save_path + image.rsplit('/',2)[1]+'/'+'idx'+str(idx)+'__'+file_name_curr.replace('.png','.pickle')

            data_curr = [[], [], [], []]
            data_curr[0] = vertices[idx]
            data_curr[1] = edges[idx]
            data_curr[2] = slide_label
            data_curr[3] = slide_idx
            data_curr = np.asarray(data_curr, dtype = object)
            if not os.path.exists(save_path_curr.rsplit('/',1)[0]):
                os.makedirs(save_path_curr.rsplit('/',1)[0])
            with open(save_path_curr,"wb") as file_pickle:
                pickle.dump(data_curr, file_pickle)
            count += 1
            print(count, image)
            # pdb.set_trace()
            # graphs.append(Data(x=vertices[idx], edge_index=edges[idx], y=slide_label))

        


        # if not os.path.exists(save_path_curr.rsplit('/',1)[0]):
        #     os.makedirs(save_path_curr.rsplit('/',1)[0])
        # with open(save_path_curr,"wb") as file_pickle:
        #     pickle.dump(data_curr, file_pickle)

        # print(image,count, A.shape,V.shape)









# address in himalaya ("/home/Drive3/Karman/TCGA_dataset/TCGA-A1-A0SD-01Z-00-DX1.DB17BFA9-D951-42A8-91D2-F4C2EBC6EB9F/*.png")
# dataset = SiamDataset(img_file_list, mode='create', detailed=True)