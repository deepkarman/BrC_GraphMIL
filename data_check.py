import numpy as np
import os, sys, glob
import pdb
import pickle
from itertools import chain

class GraphDataset(Dataset):
    def __init__(self, pickle_files_dir, file_list):
        self.pickle_files_dir=pickle_files_dir
        self.file_list = file_list
        

    def __getitem__(self, index):
        file_name = self.file_list[index]
        
        # if self.augment:
            # TODO
            
        # print('\n\n', file_name, '\n\n')
        data = pickle.load(open(file_name,'rb'))

        vertex = torch.tensor(data[0], dtype=torch.float32)

        edge = torch.tensor(data[1], dtype=torch.long)
        edge, _ =  torch_geometric.utils.add_remaining_self_loops(edge)
        
        
        l = torch.tensor(data[2])

        slide_idx = torch.tensor(data[3], dtype=torch.int)
        
        data_ =  Data(x=vertex, edge_index=edge, y=l)
        data_.slide_idx = slide_idx

        return data_

    def __len__(self):
        return len(self.file_list)


file_list = sorted(glob.glob('data_graphs/*/*'))
pickle_files_dir = 'data_graphs/'

dataset = GraphDataset('data_graphs/', file_list)

total_count = 0
directed_count = 0
num_edges = 0

for data in dataset:
	total_count += 1
	num_edges += data['edge_index'].size()[1]
	if not torch_geometric.utils.is_undirected(data['edge_index']):
		directed_count += 1
		print('this is not undirected')
		pdb.set_trace()
	pdb.set_trace()


print('avg edges are: ', (0.5*num_edges)/total_count)
pdb.set_trace()
