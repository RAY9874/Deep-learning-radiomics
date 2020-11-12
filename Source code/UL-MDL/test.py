# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from config import Config
from data_loader import get_loader
from model import GCN
from config import Config
import numpy as np
import os
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


def test():
    device = torch.device('cuda:0')
    print('using device:',device)
    config = Config()
    test_loader = get_loader(config,config.TEST_LIST, 2)
    model = GCN().to(device)
    criterion0 = nn.CrossEntropyLoss()
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()
    criterion4 = nn.CrossEntropyLoss()
    criterion5 = nn.CrossEntropyLoss()
    criterion6 = nn.CrossEntropyLoss()
    criterion7 = nn.CrossEntropyLoss()
    criterion8 = nn.CrossEntropyLoss()
    criterion9 = nn.CrossEntropyLoss()
    criterion10 = nn.CrossEntropyLoss()
    '''
    RES152 'model--epoch:10-D-acc:0.9812--L-acc:0.9932.ckpt'
    res50  'model--epoch:11-D-acc:0.9801--L-acc:0.9937.ckpt'

    '''

    for i in range(1,199):
        ckpt_path = os.path.join(config.MODEL_PATH, 'model_epoch_'+str(i)+'.ckpt')
        print('loading checkpoint from', ckpt_path)
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint)
        params = list(model.named_parameters())

        # print(params[0][1])
        data = params[0][1].detach().cpu().numpy()
        print(data.shape)
        plt.matshow(data , cmap=plt.cm.hot)
        plt.colorbar()
        plt.savefig('./Adjacent/epoch_'+str(i)+'.jpg')
        plt.close()
        del params
        del data

    exit(0)


    ckpt_path = os.path.join(config.MODEL_PATH, 'model_epoch_63.ckpt')
    print('loading checkpoint from', ckpt_path)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint)
    params = list(model.named_parameters())
    test_accs=[]
    test_loss =[]

    disease_nums = 11
    disease_names = ['主疾病','结节类型--高/等/低回声','结节类型--单/多发','边界','形态','纵横比','皮质类型','淋巴门结构','钙化','囊性区','血流']

    ################
    ###测试
    ################
    label_preds=[[] for _ in range(disease_nums)]
    label_gts = [[] for _ in range(disease_nums)]

    print('begin to predict')
    with torch.no_grad():
        for ii, (inputs,gt_labels) in enumerate(test_loader):
            print(str(ii))
            for i in range(len(inputs)):
                # _input = _input.to(torch.float)
                inputs[i] = inputs[i].type(torch.FloatTensor)
                inputs[i] = inputs[i].to(device)

            # for _input in inputs:
            #     print(_input.device)

            for i in range(len(gt_labels)):
                gt_labels[i] = gt_labels[i].to(torch.long)
                gt_labels[i] = gt_labels[i].to(device)

            pred_outputs = model(inputs[0],inputs[1])


            # loss0 = criterion0(pred_outputs[0],gt_label[0])
            # loss1 = criterion1(pred_outputs[1],gt_label[1])
            # loss2 = criterion2(pred_outputs[2],gt_label[2])
            # loss3 = criterion3(pred_outputs[3],gt_label[3])
            # loss4 = criterion4(pred_outputs[4],gt_label[4])
            # loss5 = criterion5(pred_outputs[5],gt_label[5])
            # loss6 = criterion6(pred_outputs[6],gt_label[6])
            # loss7 = criterion7(pred_outputs[7],gt_label[7])
            # loss8 = criterion8(pred_outputs[8],gt_label[8])
            # loss9 = criterion9(pred_outputs[9],gt_label[9])
            # loss10 = criterion10(pred_outputs[10],gt_label[10])

            # loss = 10*loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10


            # 记录当前的lost以及batchSize数据对应的分类准确数量
            for i in range(disease_nums):
                print(pred_outputs[i])
                _, predict = torch.max(pred_outputs[i], 1)
                print(predict)
                predict = list(predict.cpu().numpy())
                label = list(gt_labels[i].cpu().numpy())
                #存储
                label_preds[i] +=predict
                label_gts[i] +=label
  
########################
### 评价矩阵
########################
    for i in range(disease_nums):
        print('-' * 40 + '↓disease↓' + '-' * 40)
        accuracy = accuracy_score(label_preds[i], label_gts[i])
        precision = precision_score(label_preds[i], label_gts[i], average='macro')
        recall = recall_score(label_preds[i], label_gts[i], average='macro')
        f1 = f1_score(label_preds[i], label_gts[i], average='macro')

        print(' {} : accuracy:{:.4f}, precision:{:.4f}, recall:{:.4f},f1:{:.4f}'.format(
                disease_names[i],accuracy,precision,recall, f1))

    #主任务的混淆矩阵
    from sklearn.metrics import confusion_matrix
    print('confusion_matrix: ')
    for i, row in enumerate(confusion_matrix(label_preds[0], label_gts[0])):
        print(str(i) + '\t' + str(row))

    con_matrix = confusion_matrix(label_preds[0], label_gts[0])
    class_num = len(con_matrix)

    print('classnum',class_num)
    for i in range(class_num):
        print(config.DISEASE_LABELS[i],' acc :{:.4f}'.format(con_matrix[i][i] / np.sum(con_matrix[i])))
    print('-' * 80)

if __name__ == '__main__':
    test()