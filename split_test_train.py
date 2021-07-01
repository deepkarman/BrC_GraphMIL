import numpy as np
import os, sys, glob
import pdb
import pickle
from itertools import chain


list_folders = sorted(glob.glob('data_graphs/*'))
list_files = []
nums = []
count = 0


for folder in list_folders:
    files = sorted(glob.glob(folder+'/*'))
    l = pickle.load(open(files[0],'rb'))[2]
    nums.append([count, len(files), l])
    count += 1
    list_files.append(files)

nums = np.array(nums)

label_1 = nums[np.where(nums[:,-1] == 1)]
label_0 = nums[np.where(nums[:,-1] == 0)]

assert (nums.shape[0] == (label_1.shape[0]+label_0.shape[0]))

print("There are ", label_1.shape[0], " slides with N0M0 (label 1), total ", np.sum(label_1[:,1]), " graphs.")
print("There are ", label_0.shape[0], " slides with Rest (label 0), total ", np.sum(label_0[:,1]), " graphs.")
print("1 forms ", np.sum(label_1[:,1])/(np.sum(label_1[:,1])+np.sum(label_0[:,1])), " fraction of dataset")

# initial split with seed 2
for sed in range(0, 10):
	print(type(sed))
	np.random.seed(sed)

	np.random.shuffle(label_1)
	np.random.shuffle(label_0)

	train_0 = label_0[:int(0.8*label_0.shape[0]+1),:]
	test_0 = label_0[int(0.8*label_0.shape[0]+1):,:]

	train_1 = label_1[:int(0.8*label_1.shape[0]+1),:]
	test_1 = label_1[int(0.8*label_1.shape[0]+1):,:]

	num_train_1 = np.sum(train_1[:,1])
	num_train_0 = np.sum(train_0[:,1])
	num_train = num_train_1 + num_train_0

	num_test_1 = np.sum(test_1[:,1])
	num_test_0 = np.sum(test_0[:,1])
	num_test = num_test_1 + num_test_0
	print('seed is: ', sed)
	print("Total train fraction split is ", num_train/(num_train+num_test))
	print("Split within train is ", num_train_1/(num_train_1+num_train_0))
	print("Split within test is ", num_test_1/(num_test_1+num_test_0), '\n')

	if sed == 2:

		list_train = []
		for idx in range(train_0.shape[0]):
		    list_train.append(sorted(glob.glob(list_folders[train_0[idx, 0]]+'/*')))
		for idx in range(train_1.shape[0]):
		    list_train.append(sorted(glob.glob(list_folders[train_1[idx, 0]]+'/*')))
		list_train = list(chain(*list_train))

		list_test = []
		for idx in range(test_0.shape[0]):
		    list_test.append(sorted(glob.glob(list_folders[test_0[idx, 0]]+'/*')))
		for idx in range(test_1.shape[0]):
		    list_test.append(sorted(glob.glob(list_folders[test_1[idx, 0]]+'/*')))
		list_test = list(chain(*list_test))


		train_file = 'test_train_split/train_file_seed_{}.pickle'.format(sed)
		test_file = 'test_train_split/test_file_seed_{}.pickle'.format(sed)
		with open(train_file,"wb") as file_pickle:
		    pickle.dump(list_train, file_pickle)
		with open(test_file,"wb") as file_pickle:
		    pickle.dump(list_test, file_pickle)


pdb.set_trace()
