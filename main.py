import argparse
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import os, sys, glob
import sklearn
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve
from collections import Counter
import torch_geometric.transforms as T
import torch_geometric.nn as geo_nn
# from torch.utils.data import Dataset
from torch_geometric.data import Data, Dataset, DataLoader
from tensorboardX import SummaryWriter
from layers import *
import math
import pdb, time
from itertools import chain
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"
writer = SummaryWriter()
best_auc_fold = 0

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
        # V = torch.tensor(data[0][0], dtype=torch.float32)
        # V[:,1:4] = V[:,1:4]/255
        # V[:,0] = V[:,0]/10

        # A_coo = torch.tensor(data[1][0], dtype=torch.long)
        # edge_feature = torch.tensor(data[2][0], dtype=torch.float32)
        # A_coo, edge_feature =  torch_geometric.utils.add_remaining_self_loops(A_coo, edge_weight = edge_feature)

        # l = torch.tensor(data[3][0])
        
        return Data(x=vertex, edge_index=edge, y=l)

    def __len__(self):
        return len(self.file_list)







def main():
    # ------------------- Create args ----------------------
    global best_auc_fold
    args = {}
    args['batch_size'] = 20
    args['test_batch_size'] = 20
    args['no_cuda']=False
    args['seed']=7
    args['lr']=0.01
    # args['momentum']=0.5
    args['log_interval']=20
    args['epochs']=25 # epochs per fold
    args['num_iter']=1
    args['start_epoch']=1
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.manual_seed(args['seed'])
    device = torch.device("cuda:0" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    no_of_folds = 1
    # args['resume']=['checkpoints_TCGA/discard_TCGA_split2_initGCN5_noVGG_topK_.pth.tar' for fold in range(no_of_folds)]
    # args['resume']=['checkpoints_TCGA/TCGA_run_54params_topK_initBN_folds_0.pth.tar_auc_0.7935537982266956_new.pth.tar']
    acccuracies = []
    max_accuracies = []
    
    best_auc_fold = 0

    # ----------------------------Create dataset ----------------------------------------------

    train_file_list = pickle.load(open('test_train_split/train_file_seed_2.pickle','rb')) 
    test_file_list = pickle.load(open('/test_train_split/test_file_seed_2.pickle','rb'))

    dataset_train = GraphDataset('data_graphs/', train_file_list)
    dataset_test = GraphDataset('data_graphs/', test_file_list)

    train_loader = DataLoader(dataset = dataset_train, batch_size = args['batch_size'], shuffle = True, **kwargs) # no collate req; drop_last?; kwarg? 
    test_loader = DataLoader(dataset = dataset_test, batch_size = args['test_batch_size'], shuffle = True, **kwargs)

    
    # ---------------------Create model and other primitives --------------------------------------

    model = Net().float() # model at float32 precision
    # batchnorm needs accumulators to be big else they'll overflow
    # so set those to float32
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.float32()
    model = model.to(device)
    print(model)
    pdb.set_trace()
    # model = nn.DataParallel(model, device_ids=[1, 2]).to(device)

    # optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'],weight_decay=0)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    #optimizer = optim.Adam(model.parameters(), lr=args['lr'],weight_decay=0)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5 , last_epoch=-1)

    # keep loss as float32 else it will overflow
    loss_ce = nn.CrossEntropyLoss(weight = torch.tensor([0.547, 0.453],  dtype = torch.float32, device = device), reduction = 'sum') # Train on TCGA
    # loss_ce = nn.CrossEntropyLoss(weight = torch.tensor([38.0/49.0,(11.0/49.0)],  dtype = torch.float32, device = device), reduction = 'sum') # train on JH
    

    AUC_data = []
    y_true = []
    y_score = []
    fpr, tpr, thresholds_auc = None, None, None
    best_JH_auc = 0.
    for fold in range(no_of_folds):
        
        
        # Moved a bunch of stuff outside the loop of number of folds both above and below
        # Confirm this method of k fold cv, plus if other things should be moved
        # Where to calc TPR and FPR for ROC and AUC

        acc_list=[]
        if args['resume'] is not None:
            if os.path.isfile(args['resume'][fold]):
                print("=> loading checkpoint '{}'".format(args['resume'][fold]))
                checkpoint = torch.load(args['resume'][fold])
                args['start_epoch'] = checkpoint['epoch']
                # best_prec1 = checkpoint['best_prec1']
                # print("best_prec is ", best_prec1)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # scheduler.load_state_dict(checkpoint['scheduler'])
                acc, auc = calc_acc_and_auc(model, device, test_loader, loss_ce, plot=True)
                print('acc test is {:.6f}, auc on test is {:.2f}'.format(acc, auc))
                acc, auc = calc_acc_and_auc(model, device, test_loader_JH, loss_ce, plot=True)
                print('acc JH is {:.6f}, auc on JH is {:.2f}'.format(acc, auc))
                sys.exit(0)
            else:
                print("=> no checkpoint found at '{}'".format(args['resume'][fold]))
                # sys.exit(0)
        
        for epoch in range(args['start_epoch'], args['epochs'] + 1):
            y_true = []
            y_score = []
            
            train(args, model, device, loss_ce, train_loader, optimizer, epoch, test_loader, test_loader_JH, fold)

            scheduler.step()
            
            acc, auc = calc_acc_and_auc(model, device, test_loader_JH, loss_ce)
            print('Train Epoch: {} \tPresent AUC on JH: {:.3f} \t Test Acc on JH: {:.6f}'.format(epoch, auc, acc))

            # if auc > 0.65 and auc > best_JH_auc:
            #     best_JH_auc = auc
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #         'auc' : auc,
            #         }, args['resume'][fold]+'_JH_auc_'+'{:.4f}'.format(auc)+'_new.pth.tar')
            #     print('Saving best JH AUC: '+ str(auc))



        # torch.save({
        #     'epoch': epoch + 1,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        #     }, args['resume'][fold])
     #   break # only training for 1 fold
    # print('AUC is: ', AUC_data)
    del model
    torch.cuda.empty_cache()

    print("Best auc: ", best_auc_fold)

    for k, v in args.items():
        print(k, v)


if __name__ == '__main__':
    main()