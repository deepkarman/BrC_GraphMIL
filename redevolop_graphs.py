import numpy as np
import os, sys, glob
import pdb
import pickle
from itertools import chain


list_folders = sorted(glob.glob('data_graphs/*'))
list_files = []
nums = []
count = 0

idx_slide = 0

for folder in list_folders:
	files = sorted(glob.glob(folder+'/*'))
	for file in files:
		with open(file, "rb") as curr_file:
			data_obj = pickle.load(curr_file)
		data_obj.append(idx_slide)
		with open(file,"wb") as file_pickle:
			pickle.dump(data_obj, file_pickle)
	idx_slide += 1



pdb.set_trace()
