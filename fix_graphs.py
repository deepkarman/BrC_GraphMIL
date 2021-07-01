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



file_list = sorted(glob.glob('data_graphs/*/*'))

dist_thresh = 0.72

count = 0

for file in file_list:
    data_obj = None
    with open(file, "rb") as curr_file:
        data_obj = pickle.load(curr_file)
    vertex= data[0]
    l = data[2]
    slide_idx = data[3]

    dist = torch.cdist(vertex, vertex, 2)
    i, j = np.where((dist<dist_thresh))
    edge = np.array([i,j])

    data_obj[1] = edge

    with open(file,"wb") as file_pickle:
        pickle.dump(data_obj, file_pickle)

    if count%200 == 0:
        print('count: ', count)

