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

        slide_idx = torch.tensor(data[3], dtype=torch.int)
        # V = torch.tensor(data[0][0], dtype=torch.float32)
        # V[:,1:4] = V[:,1:4]/255
        # V[:,0] = V[:,0]/10

        # A_coo = torch.tensor(data[1][0], dtype=torch.long)
        # edge_feature = torch.tensor(data[2][0], dtype=torch.float32)
        # A_coo, edge_feature =  torch_geometric.utils.add_remaining_self_loops(A_coo, edge_weight = edge_feature)

        # l = torch.tensor(data[3][0])
        
        data_ =  Data(x=vertex, edge_index=edge, y=l)
        data_.slide_idx = slide_idx

        return data_

    def __len__(self):
        return len(self.file_list)




# ---------------------------- NETWORK --------------------------


class Net_torch_geo(nn.Module):
    def __init__(self):
        super(Net_torch_geo, self).__init__()

        # Conv Layer candidates
        # AGNNConv
        # ARMAConv
        # DynamicEdgeConv
        # GATConv
        # GCNConv - GINConv better?
        # SAGEConv
        # SGConv

        # Norm Later candidates
        # BatchNorm - batch normalization over a batch of node features
        # InstanceNorm - instance normalization over each individual example in a batch of node features

        # Pooling
        # EdgePooling - needs an edge score method
        # SAGPooling - needs a GNN for calculating projection scores
        # TopKPooling - better than SAGPool
        # nearest
        # radius

        # Global Pool to get graph level output

        # TODO
        # First input layer - normalise size ... how?
        # Node feature conv - reduce number from 438 to more managable
        # Edge conv? Edge features? Edge scores from incident nodes? - batchwise too

        # self.GCNConv1 = geo_nn.GCNConv(438,512, add_self_loops = False, normalize = True, improved = True, cached = False, bias = True)
        # self.bn1 = nn.BatchNorm1d(54)
        # self.GINConv1 = geo_nn.GINConv(nn.Sequential(nn.Linear(54, 64), nn.ReLU(), nn.Linear(64, 64)))
        # self.GINConv2 = geo_nn.GINConv(nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128)))
        self.GCNConv1 = geo_nn.GCNConv(512,256, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        self.GCNConv2 = geo_nn.GCNConv(256,128, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.GCNConv5 = geo_nn.GCNConv(128,128, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.gc1 = geo_nn.GraphConv(54, 64)
        # self.gc2 = geo_nn.GraphConv(64, 128)
        # self.gcn1 = GraphCNNLayer_geo(54,64,1)
        # self.gcn2 = GraphCNNLayer_geo(64,128,1)

        # self.bn1 = nn.BatchNorm1d(128)

        # self.SAGPool1 = geo_nn.SAGPooling(128, ratio=0.5) # 586+438=1024
        self.TopKPool1 = geo_nn.TopKPooling(128, ratio=0.5)
        # self.EdgePool1 = geo_nn.EdgePooling(128) # ratio is always 0.5

        # self.GCNConv2 = geo_nn.GCNConv(512, 256, add_self_loops = False, normalize = True, improved = True, cached = False, bias = True)
        
        # self.GINConv3 = geo_nn.GINConv(nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 256)))
        # self.GINConv4 = geo_nn.GINConv(nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)))
        self.GCNConv3 = geo_nn.GCNConv(128,128, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        self.GCNConv4 = geo_nn.GCNConv(128,128, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.GCNConv6 = geo_nn.GCNConv(256,256, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.gc3 = geo_nn.GraphConv(128, 256)
        # self.gc4 = geo_nn.GraphConv(256, 256)
        # self.gcn3 = GraphCNNLayer_geo(128,256,1)
        # self.gcn4 = GraphCNNLayer_geo(256,256,1)

        # self.bn2 = nn.BatchNorm1d(256)

        # self.SAGPool2 = geo_nn.SAGPooling(512, ratio=0.5) # 1024+256=1280
        # self.TopKPool2 = geo_nn.TopKPooling(256, ratio=0.5)

        # self.GCNConv3 = geo_nn.GCNConv(256, 256, add_self_loops = False, normalize = True, improved = True, cached = False, bias = True)
        
        # self.GINConv5 = geo_nn.GINConv(nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)))
        # self.GINConv6 = geo_nn.GINConv(nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)))
        # self.GCNConv5 = geo_nn.GCNConv(256,256, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.GCNConv6 = geo_nn.GCNConv(256,256, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)

        # self.bn3 = nn.BatchNorm1d(256)

        # self.SAGPool3 = geo_nn.SAGPooling(256, ratio=0.5)
        self.TopKPool3 = geo_nn.TopKPooling(128, ratio=0.5)
        # self.EdgePool3 = geo_nn.EdgePooling(256) # ratio is always 0.5

        # self.GINConv7 = geo_nn.GINConv(nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256)))
        self.GCNConv7 = geo_nn.GCNConv(128,128, add_self_loops = True, normalize = True, improved = True, cached = False, bias = True)
        # self.gc7 = geo_nn.GraphConv(256, 256)
        # self.gcn7 = GraphCNNLayer_geo(256,256,1)

        self.post_gcn = nn.Sequential(
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(), 
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), 
            nn.Linear(64,2)
            )



    def forward(self, batch, x, edge_index, slide_idx, y_target):
        # TODO

        # x = self.bn1(x)
        # x = self.GINConv2(self.GINConv1(x, edge_index), edge_index) # x of shape [num_nodes, 512] now
        # x = self.GCNConv5(self.GCNConv2(self.GCNConv1(x, edge_index), edge_index), edge_index)
        x = self.GCNConv2(self.GCNConv1(x, edge_index), edge_index)
        # x = self.gc2(self.gc1(x, edge_index), edge_index)
        # x = self.gcn1(x, edge_index, edge_attr, batch)
        # x = self.gcn2(x, edge_index, edge_attr, batch)
        # x = torch.cat((x, self.GINConv2(self.GINConv1(x, edge_index), edge_index)), dim=1) # x of shape [num_nodes, 438+586=1024] now

        # x = self.bn1(x)

        # x, edge_index, edge_attr, batch, _,_ = self.SAGPool1(x, edge_index, edge_attr=edge_attr, batch=batch)
        x, edge_index, _, batch, perm,_ = self.TopKPool1(x, edge_index, batch=batch)
        # x, edge_index, batch, _ = self.EdgePool1(x, edge_index, batch)
        
        slide_idx = slide_idx[perm]


        # x = self.GINConv4(self.GINConv3(x, edge_index), edge_index) # x of shape [num_nodes, 512] now 
        # x = self.GCNConv6(self.GCNConv4(self.GCNConv3(x, edge_index), edge_index), edge_index)
        x = self.GCNConv4(self.GCNConv3(x, edge_index), edge_index)
        # x = self.gc4(self.gc3(x, edge_index), edge_index)
        # x = self.gcn3(x, edge_index, edge_attr, batch)
        # x = self.gcn4(x, edge_index, edge_attr, batch)
        # x = torch.cat((x, self.GINConv4(self.GINConv3(x, edge_index), edge_index)), dim=1) # x of shape [num_nodes, 1024] now 
 
        # x = self.bn2(x)

        # x, edge_index, edge_attr, batch, _,_ = self.SAGPool2(x, edge_index, edge_attr=edge_attr, batch=batch)
        # x, edge_index, _, batch, perm,_ = self.TopKPool2(x, edge_index, batch=batch)

        # slide_idx = slide_idx[perm]


        # x = self.GINConv6(self.GINConv5(x, edge_index), edge_index) # x of shape [num_nodes, 512] now 
        # x = torch.cat((x, self.GINConv6(self.GINConv5(x, edge_index), edge_index)), dim=1) # x of shape [num_nodes, 1024+512] now 
        # x = self.GINConv6(self.GINConv5(x, edge_index), edge_index) # x of shape [num_nodes, 512] now
        # x = self.GCNConv6(self.GCNConv5(x, edge_index), edge_index)

        
        # x = self.bn3(x)
        
        # x, edge_index, _, batch, _,_ = self.SAGPool3(x, edge_index, edge_attr=edge_attr, batch=batch)
        x, edge_index, _, batch, perm,_ = self.TopKPool3(x, edge_index, batch=batch)
        # x, edge_index, batch, _ = self.EdgePool3(x, edge_index, batch)

        slide_idx = slide_idx[perm]

        # x = self.GINConv7(x, edge_index) # x of shape [num_nodes, 512]
        x = self.GCNConv7(x, edge_index)
        # x = self.gc7(x, edge_index)
        # x = self.gcn7(x, edge_index, edge_attr, batch)
        # x = torch.cat((x, self.GINConv7(x, edge_index)), dim=1) # x of shape [num_nodes, 2048]

        x = geo_nn.global_mean_pool(x, batch) # x of shape [1, 128] now - for each graph in batch
        # x = geo_nn.global_max_pool(x, batch) # x of shape [1, 128] now - for each graph in batch

        pdb.set_trace()

        x = self.post_gcn(x)

        return x, y_target


# --------------------------- END NETWORK ------------------------



def train(args, model, device, loss_ce, train_loader, optimizer, epoch, test_loader, fold):
    global best_auc_fold
    global best_auc_JH

    running_loss = 0.
    correct_preds = 0
    elements = 0

    for batch_idx, data_ in enumerate(train_loader):

        batch = data_['batch'].to(device)
        slide_idx = data_['slide_idx'].to(device)
        edge_index = data_['edge_index'].to(device)
        x_data = data_['x'].to(device)
        y_target = data_['y'].to(device)

        optimizer.zero_grad()

        # for idx_iter in range(args['num_iter']):
        model.train()

        output, y_target = model(batch, x_data, edge_index, slide_idx, y_target) 
        
        loss = loss_ce(output, y_target) 
        preds = output.max(1, keepdim=True)[1]
        elements += len(y_target)
        train_batch_corr = preds.eq(y_target.view_as(preds)).sum().item()
        train_batch_accuracy = (100*train_batch_corr)/len(y_target)
        correct_preds += train_batch_corr
        # pdb.set_trace()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
            
        curr_loss = loss.item()
        running_loss += loss.item()*data_.num_graphs
        idx = batch_idx + (epoch-1)*math.ceil(len(train_loader))
        writer.add_scalar('Train Loss', curr_loss, idx)  
        writer.add_scalar('Train Batch Accuracy', train_batch_accuracy, idx)
        # if epoch == 10:
        #     pdb.set_trace()

        del batch
        del slide_idx
        del edge_index
        del x_data       
        del y_target 

        torch.cuda.empty_cache()

        if batch_idx % args['log_interval'] == 0 :
            acc_test, auc_test = calc_acc_and_auc(model, device, test_loader, loss_ce)

            writer.add_scalar('Test Acc', acc_test, idx)
            writer.add_scalar('AUC', auc_test, idx)

            print('Train Epoch: {} [({:.0f}%)]\tLoss: {:.6f} \tAcc Train: {:.2f}% \tAcc test {:.2f}\tTest AUC: {:.3f}'.format(
                epoch, 100. * (batch_idx+1) / len(train_loader), curr_loss, 100.*correct_preds/elements,acc_test, auc_test))

            # if auc_test>best_auc_fold and auc_test > 0.65:
            #     best_auc_fold = auc_test
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'auc' : auc_test,
            #         }, args['resume'][fold]+'_testauc_'+str(auc_test)+'_new.pth.tar')
            #     print('Saving best AUC: '+ str(auc_test))

    running_loss /= len(train_loader.dataset)
    print("Loss for epoch {} is: {:.6f} \tAccuracy for Epoch is {:.2f}".format(epoch, running_loss, 100.*correct_preds/elements))
    writer.add_scalar('Train loss epoch', running_loss, epoch)

    return 


'''
def test(model, device, test_loader, loss_ce):
    
    test_loss = 0.
    y_score = []
    y_true  = []
    m = nn.Softmax(dim = 1)

    # start = time.time()
    for batch_idx, data_ in enumerate(test_loader):
        with torch.no_grad():
            batch = data_['batch'].to(device)
            edge_attr = data_['edge_attr'].to(device)
            edge_index = data_['edge_index'].to(device)
            x_data = data_['x'].to(device)
            y_target = data_['y'].to(device)

            model.eval()

            output = model(batch, x_data, edge_index, edge_attr) 
            test_loss += loss_ce(output, y_target).item()*data_.num_graphs # sum up batch loss
            
            y_score.append(np.array(m(output).cpu()[:,1]).tolist())
            y_true.append(np.array(y_target.cpu()).tolist())

            del batch
            del edge_attr
            del edge_index
            del x_data       
            del y_target 

            torch.cuda.empty_cache()
    # print(" Test takes ", time.time()-start," seconds")     

    try:
        y_true = list(chain(*y_true))
        y_score = list(chain(*y_score))
        auc = roc_auc_score(y_true,y_score) # TODO - check this
    except ValueError as e:
        print("auc has error")
        print(e)
        pdb.set_trace()
        auc = 0
    test_loss /= len(test_loader.dataset)

    return test_loss, auc
'''

def calc_acc_and_auc(model, device, data_loader, loss_ce, plot = False):

    y_score = []
    y_true  = []
    y_pred = []
    m = nn.Softmax(dim = 1)
    correct_preds = 0

    for batch_idx, data_ in enumerate(data_loader):
        with torch.no_grad():
            batch = data_['batch'].to(device)
            edge_index = data_['edge_index'].to(device)
            slide_idx = data_['slide_idx'].to(device)
            x_data = data_['x'].to(device)
            y_target = data_['y'].to(device)

            model.eval()

            output, y_target = model(batch, x_data, edge_index, slide_idx, y_target) 
            
            preds = output.max(1, keepdim=True)[1]
            correct_preds += preds.eq(y_target.view_as(preds)).sum().item()
            
            y_score.append(np.array(m(output).cpu()[:,1]).tolist())
            y_true.append(np.array(y_target.cpu()).tolist())
            y_pred.append(np.array(m(output).max(1,keepdim=True)[1].cpu()).tolist())

            del batch
            del slide_idx
            del edge_index
            del x_data       
            del y_target 

            torch.cuda.empty_cache()

    try:
        y_true = list(chain(*y_true))
        y_score = list(chain(*y_score))
        y_pred = list(chain(*y_pred))
        auc = roc_auc_score(y_true,y_score) # TODO - check this
        # pdb.set_trace()
    except ValueError as e:
        print("auc has error")
        print(e)
        pdb.set_trace()
        auc = 0

    if plot:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        print('showing plot')
        plt.savefig('auc_roc_curve_'+str(data_loader)+'.png')
    acc = 100.*correct_preds/len(data_loader.dataset)
    return acc, auc




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

    model = Net_torch_geo().float() # model at float32 precision
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
    loss_ce = nn.CrossEntropyLoss(weight = torch.tensor([0.972, 1.028],  dtype = torch.float32, device = device), reduction = 'sum') # Train on TCGA
    

    AUC_data = []
    y_true = []
    y_score = []
    fpr, tpr, thresholds_auc = None, None, None
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
            
            train(args, model, device, loss_ce, train_loader, optimizer, epoch, test_loader, fold)

            scheduler.step()
            
            # acc, auc = calc_acc_and_auc(model, device,, loss_ce)
            # print('Train Epoch: {} \tPresent AUC on JH: {:.3f} \t Test Acc on JH: {:.6f}'.format(epoch, auc, acc))

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