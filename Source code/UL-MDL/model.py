# -*- coding: utf-8 -*-
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchvision.models import resnet50,resnet152

from torch.nn import functional as F


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(in_features, out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(out_features, 1).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(1,2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters_uniform()

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self,nhid=1024,nout=2048):
        super(GCN, self).__init__()

        resnet_us =resnet50(pretrained=True)
        modules_us = list(resnet_us.children())[:-1]  # delete the last fc layer.
        self.resnet_us = nn.Sequential(*modules_us)

        resnet_dpl =resnet50(pretrained=True)
        modules_dpl = list(resnet_dpl.children())[:-1]  # delete the last fc layer.
        self.resnet_dpl = nn.Sequential(*modules_dpl)

        self.gc1 = GraphConvolution(2*resnet_us.fc.in_features, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = 0.1
        self.leakyrelu1 = nn.LeakyReLU(0.2)

        self.atten_conv = nn.Conv2d(in_channels=2*resnet_us.fc.in_features, out_channels=11, kernel_size=(1,1))

        _adj = np.random.uniform(0,1 ,(11,11)).astype(np.float32)

        self.A = Parameter(torch.from_numpy(_adj).float())

        #主任务
        self.linear1 = nn.Linear(nout, 3)
        self.bn1 = nn.BatchNorm1d(3, momentum=0.01)
        # jiejieleixing_huisheng,#结节类型--高/等/低回声 3种
        self.linear2 = nn.Linear(nout, 3)
        self.bn2 = nn.BatchNorm1d(3, momentum=0.01)
        # jiejieleixing_fa, 结节类型--单/多发 2种
        self.linear3 = nn.Linear(nout, 2)
        self.bn3 = nn.BatchNorm1d(2, momentum=0.01)
        # bianjie, 边界 清/不清 2种
        self.linear4 = nn.Linear(nout, 2)
        self.bn4 = nn.BatchNorm1d(2, momentum=0.01)
        # xingtai,形态 规则/不规则 2种
        self.linear5 = nn.Linear(nout, 2)
        self.bn5 = nn.BatchNorm1d(2, momentum=0.01)
        # zonghengbi,纵横比 >2/<2 2种
        self.linear6 = nn.Linear(nout, 2)
        self.bn6 = nn.BatchNorm1d(2, momentum=0.01)
        # pizhileixing,皮质类型 偏心/不偏心/无淋巴门 3种
        self.linear7 = nn.Linear(nout, 3)
        self.bn7 = nn.BatchNorm1d(3, momentum=0.01)
        # linbamenjiegou,淋巴门结构 有/无 2种
        self.linear8 = nn.Linear(nout, 2)
        self.bn8 = nn.BatchNorm1d(2, momentum=0.01)
        # gaihua,钙化 无/点状/粗大 3种
        self.linear9 = nn.Linear(nout, 3)
        self.bn9 = nn.BatchNorm1d(3, momentum=0.01)
        # nangxingqu, #囊性区 有/无 2种
        self.linear10 = nn.Linear(nout, 2)
        self.bn10 = nn.BatchNorm1d(2, momentum=0.01) 
        # xueliu 血流 1/2/3/4/5型 5种
        self.linear11 = nn.Linear(nout, 5)
        self.bn11 = nn.BatchNorm1d(5, momentum=0.01)

    def forward(self,us_image,dpl_image):
        us_features = self.resnet_us(us_image)
        dpl_features = self.resnet_dpl(dpl_image)
        concat_features = torch.cat([us_features,dpl_features],1) # 6, 4096 ,1,1
        # print('concat_features',concat_features.shape)
        att_features = self.atten_conv(concat_features) # 6,11,1,1
        # print('att_features',att_features.shape)
        att_features_shape = att_features.shape 
        att_features= att_features.reshape(att_features_shape[0],att_features_shape[1], -1)
        att_features = F.softmax(att_features, dim=2) #6, 11,

        # print('att_features',att_features.shape)
        att_features= att_features.reshape(att_features_shape)#6, 11,1,1
        # print('att_features',att_features.shape)
        gcn_inputs = []
        for i in range(11):
            temp = torch.mul(att_features[:,i:i+1,:,:] ,concat_features)# 1,4096,1,1
            temp = torch.sum(temp,2)# 1,4096,1
            temp = torch.sum(temp,2,keepdim=True)# 1,4096,1
            gcn_inputs.append(temp)
        gcn_inputs = torch.cat(gcn_inputs,2) # 6,4096,11,


        gcn_inputs= gcn_inputs.reshape(gcn_inputs.shape[0],gcn_inputs.shape[2],gcn_inputs.shape[1])#论文中是 class * demension
        # print('self.A.shape',self.A.shape)
        # print('gcn_inputs',gcn_inputs.shape)
        gcn1 = self.gc1(gcn_inputs, self.A)
        gcn1 = self.leakyrelu1(gcn1)
        gcn2 = self.gc2(gcn1, self.A)

        # print(gcn2.shape)
        features1 = self.bn1(self.linear1(gcn2[:,0,:]))
        features2 = self.bn2(self.linear2(gcn2[:,1,:]))
        features3 = self.bn3(self.linear3(gcn2[:,2,:]))
        features4 = self.bn4(self.linear4(gcn2[:,3,:]))
        features5 = self.bn5(self.linear5(gcn2[:,4,:]))
        features6 = self.bn6(self.linear6(gcn2[:,5,:]))
        features7 = self.bn7(self.linear7(gcn2[:,6,:]))
        features8 = self.bn8(self.linear8(gcn2[:,7,:]))
        features9 = self.bn9(self.linear9(gcn2[:,8,:]))
        features10 = self.bn10(self.linear10(gcn2[:,9,:]))
        features11 = self.bn11(self.linear11(gcn2[:,10,:]))

        return features1,features2,features3,features4,features5,features6,features7,features8,features9,features10,features11


