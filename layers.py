import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pickle
import numpy as np
import os
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import time
import torch_geometric as torch_geo
import pdb

class GraphCNNLayer_geo(nn.Module):
    def __init__(self, input_features_V, output_features_V, no_A, is_relu=True, is_bn=True):
        super(GraphCNNLayer_geo, self).__init__()

        self.lin_A = nn.Linear(input_features_V*no_A, output_features_V, bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5) 

        self.lin_I = nn.Linear(input_features_V,output_features_V,bias=False)
        self.lin_I.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5)

        self.bn = nn.BatchNorm1d(output_features_V)
        self.relu = torch.nn.ReLU()
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(output_features_V)

    def forward(self, x, edge_index, edge_attr, batch):
        # x: (batch*N) x F
        # V: batch x N x F
        V, V_label = torch_geo.utils.to_dense_batch(x, batch=batch)

        # A: batch x NL x N
        A = torch_geo.utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr)

        result_V = torch.add(self.lin_A(torch.matmul(A,V)),self.lin_I(V))

        if(self.is_relu):
            result_V = self.relu(result_V)
        F = result_V.shape[2]
        result_x = result_V.view(-1,F)[V_label.flatten(),:]

        if(self.is_bn):
            result_x = self.bn(result_x)

        return result_x


class GraphCNNLayer(nn.Module):
    def __init__(self, input_features_V, output_features_V, no_A, is_relu=True, is_bn=True):
        super(GraphCNNLayer, self).__init__()

        self.lin_A = nn.Linear(input_features_V*no_A, output_features_V, bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5) 

        self.lin_I = nn.Linear(input_features_V,output_features_V,bias=False)
        self.lin_I.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5)

        self.bn = nn.BatchNorm1d(output_features_V)
        self.relu = torch.nn.ReLU()
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(output_features_V)
        

    def forward(self,A,V): 
        # A:batchxNxLxN 
        # V:batchxNxF        
        # Concatenate the adj matrices
        L = A.shape[2]#=1
        N = A.shape[1]
        f = V.shape[-1]
        A_reshape = A.view(-1,N*L,N).float()#(batchxNLxN)
        n = torch.matmul(A_reshape,V)#(batchxNLxF)
        n = n.view(-1,N,L*f)#(batchxNxLF)
        
        result_V = torch.add(self.lin_A(n),self.lin_I(V))
        if(self.is_relu):
            result_V = self.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_V


class GraphEmbedPooling2_geo(nn.Module):
    # TODO
    def __init__(self, input_features_V, output_features_V, no_A, is_relu=True, is_bn=True):
        super(GraphEmbedPooling2, self).__init__()

        #self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)

        self.lin_A = nn.Linear(input_features_V, output_features_V, bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5) 

        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_relu):
            self.relu = nn.ReLU()
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self, x, edge_index, edge_attr, batch): 
        # x: (batch*N) x F
        # V: batch x N x F
        V, V_label = torch_geo.utils.to_dense_batch(x, batch=batch)

        # A: batch x NL x N
        A = torch_geo.utils.to_dense_adj(edge_index, batch=batch, edge_attr=edge_attr)

        V_emb = F.softmax(self.lin_A(V.float()),dim = 1)#(batchxNxN')
        N1 = V_emb.shape[-1]
        result_V = torch.matmul(V_emb.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = self.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)

        raise Exception('Method not implemented yet - fix code first')
        return x, edge_index, edge_attr, batch,



class GraphEmbedPooling2(nn.Module): #Zerohop
    # TODO - adapt this
    def __init__(self, input_features_V, output_features_V, no_A, is_relu=True, is_bn=True):
        super(GraphEmbedPooling2, self).__init__()

        #self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)

        self.lin_A = nn.Linear(input_features_V, output_features_V, bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V*(no_A+1))**0.5) 

        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_relu):
            self.relu = nn.ReLU()
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        V_emb = F.softmax(self.lin_A(V.float()),dim = 1)#(batchxNxN')
        N1 = V_emb.shape[-1]
        result_V = torch.matmul(V_emb.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = self.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V




class GraphCNNLayer2(nn.Module):
    def __init__(self,input_features_V,no_A_in,no_A_out,is_relu=True,is_bn=False,bn_after_relu=False):
        super(GraphCNNLayer2, self).__init__()
        self.lin = nn.Linear(no_A_in+2*input_features_V,no_A_out,bias=True)
        self.lin.bias.data.fill_(0.1) 
        self.lin.weight.data.normal_(0,1.0/(no_A_out*(no_A_in+2*input_features_V))**0.5) 
        self.bn = nn.BatchNorm1d(no_A_out)
        self.is_relu = is_relu
        self.is_bn = is_bn
        self.bn_after_relu = bn_after_relu
        if(self.is_bn):
            self.bn = nn.BatchNorm2d(no_A_out)
        

    def forward(self,A,V):#Change no of edge features only 
        # A:batchxNxLxN 
        # V:batchxNxF        
        L = A.shape[2]#=1
        N = A.shape[1]
        f = V.shape[-1]
        b=V.shape[0]
    
        V1=torch.unsqueeze(V,1)#bxNxF=>bx1xNxF
        V2=torch.unsqueeze(V,2)#bxNxF=>bxNx1xF
        V1=V1.expand(b,N,N,f)
        V2=V2.expand(b,N,N,f)
        # print(V1.shape)
        A1 = A.transpose(2,3)#bxNxLxN=>bxNxNxL
        # print(A1.shape,V1.shape,V2.shape)
        A1 = torch.cat((A1,V1,V2),dim = 3)
        # print(A1.shape)
        Aout = self.lin(A1)#bxNxNxL=>bxNxNxL'
        Aout = Aout.transpose(2,3)#bxNxNxL'=>bxNxL'xN
        # print(Aout.shape)
        if not self.bn_after_relu:
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
            if(self.is_relu):
                Aout = F.relu(Aout)
        else:
            if(self.is_relu):
                Aout = F.relu(Aout)
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
        return Aout.contiguous()

class GraphCNNLayer2ReLu6(nn.Module):
    def __init__(self,input_features_V,no_A_in,no_A_out,is_relu=True,is_bn=False,bn_after_relu=False):
        super(GraphCNNLayer2ReLu6, self).__init__()
        self.lin = nn.Linear(no_A_in+2*input_features_V,no_A_out,bias=True)
        self.lin.bias.data.fill_(0.1) 
        self.lin.weight.data.normal_(0,1.0/(no_A_out*(no_A_in+2*input_features_V))**0.5) 
        self.bn = nn.BatchNorm1d(no_A_out)
        self.is_relu = is_relu
        self.is_bn = is_bn
        self.bn_after_relu = bn_after_relu
        if(self.is_bn):
            self.bn = nn.BatchNorm2d(no_A_out)
        

    def forward(self,A,V):#Change no of edge features only 
        # A:batchxNxLxN 
        # V:batchxNxF        
        L = A.shape[2]#=1
        N = A.shape[1]
        f = V.shape[-1]
        b=V.shape[0]
        V1=torch.unsqueeze(V,1)#bxNxF=>bx1xNxF
        V2=torch.unsqueeze(V,2)#bxNxF=>bxNx1xF
        V1=V1.expand(b,N,N,f)
        V2=V2.expand(b,N,N,f)
        A1 = A.transpose(2,3)#bxNxLxN=>bxNxNxL
        A1 = torch.cat((A1,V1,V2),dim = 3)
        Aout = self.lin(A1)#bxNxNxL=>bxNxNxL'
        Aout = Aout.transpose(2,3)#bxNxNxL'=>bxNxL'xN
        if not self.bn_after_relu:
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
            if(self.is_relu):
                Aout = F.relu6(Aout)
        else:
            if(self.is_relu):
                Aout = F.relu6(Aout)
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
        return Aout.contiguous()

class GraphCNNLayer2Sigmoid(nn.Module):
    def __init__(self,input_features_V,no_A_in,no_A_out,is_relu=True,is_bn=False,bn_after_relu=False):
        super(GraphCNNLayer2Sigmoid, self).__init__()
        self.lin = nn.Linear(no_A_in+2*input_features_V,no_A_out,bias=True)
        self.lin.bias.data.fill_(0.1) 
        self.lin.weight.data.normal_(0,1.0/(no_A_out*(no_A_in+2*input_features_V))**0.5) 
        self.bn = nn.BatchNorm1d(no_A_out)
        self.is_relu = is_relu
        self.is_bn = is_bn
        self.bn_after_relu = bn_after_relu
        if(self.is_bn):
            self.bn = nn.BatchNorm2d(no_A_out)
        

    def forward(self,A,V):#Change no of edge features only 
        # A:batchxNxLxN 
        # V:batchxNxF        
        L = A.shape[2]#=1
        N = A.shape[1]
        f = V.shape[-1]
        b=V.shape[0]
        V1=torch.unsqueeze(V,1)#bxNxF=>bx1xNxF
        V2=torch.unsqueeze(V,2)#bxNxF=>bxNx1xF
        V1=V1.expand(b,N,N,f)
        V2=V2.expand(b,N,N,f)
        A1 = A.transpose(2,3)#bxNxLxN=>bxNxNxL
        A1 = torch.cat((A1,V1,V2),dim = 3)
        Aout = self.lin(A1)#bxNxNxL=>bxNxNxL'
        Aout = Aout.transpose(2,3)#bxNxNxL'=>bxNxL'xN
        if not self.bn_after_relu:
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
            if(self.is_relu):
                Aout = F.sigmoid(Aout)
        else:
            if(self.is_relu):
                Aout = F.sigmoid(Aout)
            if(self.is_bn):
                Aout = Aout.transpose(1,2)
                Aout = self.bn(Aout.contiguous())
                Aout = Aout.transpose(1,2)
        return Aout.contiguous()

class GraphCNNLayer2_1(nn.Module):#Use gaussian features
    def __init__(self,input_features_V,no_A_in,no_A_out,is_relu=False,is_bn=False):
        super(GraphCNNLayer2_1, self).__init__()
        self.lin = nn.Linear(no_A_in+1,no_A_out,bias=True)
        self.lin.bias.data.fill_(0.1) 
        self.lin.weight.data.normal_(0,1.0/(no_A_out*(no_A_in+1))**0.5) 
        self.bn = nn.BatchNorm1d(no_A_out)
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm2d(no_A_out)
        
    def forward(self,A,V):#Change no of edge features only 
        # A:batchxNxLxN 
        # V:batchxNxF        
        L = A.shape[2]#=1
        N = A.shape[1]
        f = V.shape[-1]
        b = V.shape[0]
        V1=torch.unsqueeze(V,1)#bxNxF=>bx1xNxF
        V2=torch.unsqueeze(V,2)#bxNxF=>bxNx1xF
        V1=V1.expand(b,N,N,f)
        V2=V2.expand(b,N,N,f)
        dif =torch.pow(V1-V2,2)#bxNxNxf
        dif=torch.sum(dif, dim=3, keepdim=True)
        gaus = torch.exp(-dif)
        A1 = A.transpose(2,3)#bxNxLxN=>bxNxNxL
        A1 = torch.cat((A1,gaus),dim = 3)
        Aout = self.lin(A1)#bxNxNxL=>bxNxNxL'
        Aout = Aout.transpose(2,3)#bxNxNxL'=>bxNxL'xN
        if(self.is_relu):
            Aout = F.relu(Aout)
        if(self.is_bn):
            Aout = Aout.transpose(1,2)
            Aout = self.bn(Aout.contiguous())
            Aout = Aout.transpose(1,2)
        return Aout.contiguous()

class GraphEmbedPooling(nn.Module):
    def __init__(self,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):#output_features_V=output_no_of_nodes
        super(GraphEmbedPooling, self).__init__()
        self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)
        
    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        V_emb = F.softmax(self.gcn_emb(A,V),dim = 1)#(batchxNxN')
        N1 = V_emb.shape[-1]
        result_V = torch.matmul(V_emb.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1])#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GraphEmbedPoolingAsym(nn.Module):
    def __init__(self,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):#output_features_V=output_no_of_nodes
        super(GraphEmbedPoolingAsym, self).__init__()
        self.gcn_emb1 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb2 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb3 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)
        
    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        V_emb1 = F.softmax(self.gcn_emb1(A,V),dim = 1)#(batchxNxN')
        V_emb2 = F.softmax(self.gcn_emb2(A,V),dim = 1)#(batchxNxN')
        V_emb3 = F.softmax(self.gcn_emb3(A,V),dim = 1)#(batchxNxN')
        N1 = V_emb1.shape[-1]
        result_V = torch.matmul(V_emb1.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1])#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb2)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb3.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GraphEmbedPoolingSymInit(nn.Module):
    def __init__(self,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):#output_features_V=output_no_of_nodes
        super(GraphEmbedPoolingSymInit, self).__init__()
        self.gcn_emb1 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb2 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb3 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb2.lin_A.weight.data = self.gcn_emb3.lin_A.weight.data
        self.gcn_emb2.lin_I.weight.data = self.gcn_emb3.lin_I.weight.data
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)
        
    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        V_emb1 = F.softmax(self.gcn_emb1(A,V),dim = 1)#(batchxNxN')
        V_emb2 = F.softmax(self.gcn_emb2(A,V),dim = 1)#(batchxNxN')
        V_emb3 = F.softmax(self.gcn_emb3(A,V),dim = 1)#(batchxNxN')
        N1 = V_emb1.shape[-1]
        result_V = torch.matmul(V_emb1.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1])#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb2)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb3.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GraphEmbedPoolingSym(nn.Module):
    def __init__(self,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):#output_features_V=output_no_of_nodes
        super(GraphEmbedPoolingSym, self).__init__()
        self.gcn_emb1 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.gcn_emb2 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        #self.gcn_emb3 = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)
        
    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        V_emb1 = F.softmax(self.gcn_emb1(A,V),dim = 1)#(batchxNxN')
        V_emb2 = F.softmax(self.gcn_emb2(A,V),dim = 1)#(batchxNxN')
        #V_emb3 = F.softmax(self.gcn_emb3(A,V),dim = 1)#(batchxNxN')
        N1 = V_emb1.shape[-1]
        result_V = torch.matmul(V_emb1.transpose(1,2),V)#(batchxN'xF)
        result_A = A.view(A.shape[0],-1,A.shape[-1])#(batchxNLxN)
        result_A = torch.matmul(result_A,V_emb2)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = torch.matmul(V_emb2.transpose(1,2),result_A)#(batchxN'xLN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V



class ZeroHopFilter(nn.Module):
    def __init__(self,input_features_V,output_features_V,is_relu=False,is_bn=False):
        super(ZeroHopFilter, self).__init__()
        self.lin_A = nn.Linear(input_features_V,output_features_V,bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V*(1+1))**0.5) 
        self.is_relu = is_relu
        self.is_bn = is_bn
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,V): 
        result_V = self.lin_A(V.float())#(batchxNxF)
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_V

class GlobalAveragePoolingASym(nn.Module):
    def __init__(self,input_nodes_V,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):
        super(GlobalAveragePoolingASym, self).__init__()
        self.output_nodes = output_features_V
        self.is_relu = is_relu
        self.is_bn = is_bn
        #self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.lin_A = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        self.lin_B = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_B.bias.data.fill_(0.1) 
        self.lin_B.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        self.lin_C = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_C.bias.data.fill_(0.1) 
        self.lin_C.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        N1 = self.output_nodes
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = self.lin_B(result_A)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = result_A.transpose(1,2)#(batchxLN'xN)
        result_A = self.lin_C(result_A)#(batchxLN'xN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        result_V = self.lin_A(V.transpose(1,2))
        result_V = result_V.transpose(1,2)
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GlobalAveragePoolingSymInit(nn.Module):
    def __init__(self,input_nodes_V,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):
        super(GlobalAveragePoolingSymInit, self).__init__()
        self.output_nodes = output_features_V
        self.is_relu = is_relu
        self.is_bn = is_bn
        self.lin_A = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        self.lin_B = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_B.bias.data.fill_(0.1) 
        self.lin_B.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        self.lin_C = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_C.bias.data.fill_(0.1) 
        self.lin_C.weight.data = self.lin_B.weight.data
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        N1 = self.output_nodes
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = self.lin_B(result_A)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = result_A.transpose(1,2)#(batchxLN'xN)
        result_A = self.lin_C(result_A)#(batchxLN'xN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        result_V = self.lin_A(V.transpose(1,2))
        result_V = result_V.transpose(1,2)
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GlobalAveragePoolingSym(nn.Module):# symmetrical for A
    def __init__(self,input_nodes_V,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):
        super(GlobalAveragePoolingSym, self).__init__()
        self.output_nodes = output_features_V
        self.is_relu = is_relu
        self.is_bn = is_bn
        #self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.lin_A = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        self.lin_B = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_B.bias.data.fill_(0.1) 
        self.lin_B.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        N1 = self.output_nodes
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = self.lin_B(result_A)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = result_A.transpose(1,2)#(batchxLN'xN)
        result_A = self.lin_B(result_A)#(batchxLN'xN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        result_V = self.lin_A(V.transpose(1,2))
        result_V = result_V.transpose(1,2)
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V

class GlobalAveragePoolingSymMax(nn.Module):# symmetrical for V and A like GEP
    def __init__(self,input_nodes_V,input_features_V,output_features_V,no_A,is_relu=True,is_bn=True):
        super(GlobalAveragePoolingSymMax, self).__init__()
        self.output_nodes = output_features_V
        self.is_relu = is_relu
        self.is_bn = is_bn
        #self.gcn_emb = GraphCNNLayer(input_features_V,output_features_V,no_A)
        self.lin_A = nn.Linear(input_nodes_V,output_features_V,bias=True)
        self.lin_A.bias.data.fill_(0.1) 
        self.lin_A.weight.data.normal_(0,1.0/(input_features_V)**0.5) 
        if(self.is_bn):
            self.bn = nn.BatchNorm1d(input_features_V)

    def forward(self,A,V): 
        no_nodes = A.shape[-1]
        N1 = self.output_nodes
        result_A = A.view(A.shape[0],-1,A.shape[-1]).float()#(batchxNLxN)
        result_A = self.lin_A(result_A)#(batchxNLxN')
        result_A = result_A.view(A.shape[0],A.shape[-1],-1)#(batchxNxLN')
        result_A = result_A.transpose(1,2)#(batchxLN'xN)
        result_A = self.lin_A(result_A)#(batchxLN'xN')
        result_A = result_A.view(A.shape[0],N1,A.shape[2],N1)#(batchxN'xLxN')
        result_V = self.lin_A(V.transpose(1,2))
        result_V = result_V.transpose(1,2)
        if(self.is_relu):
            result_V = F.relu(result_V)
        if(self.is_bn):
            result_V = result_V.transpose(1,2)
            result_V = self.bn(result_V.contiguous())
            result_V = result_V.transpose(1,2)
        return result_A,result_V
