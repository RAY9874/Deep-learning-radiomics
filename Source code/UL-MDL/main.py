import torch
import torch.nn as nn
from config import Config
from data_loader import get_loader
from model import GCN
from config import Config
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


def count_correct_nums(output,label):
    _, predict = torch.max(output, 1)
    correct_num = (predict == label).sum()
    return correct_num.item()

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def main():
    # print('gpu avaliable ' if torch.cuda.is_available() else "gpu not alvliable")
    device = torch.device("cuda:0")
    print('using device:', device)
    config = Config()
    train_loader = get_loader(config,config.TRAIN_LIST, 2)
    val_loader = get_loader(config,config.VAL_LIST, 2)
    model = GCN().to(device)
    if not os.path.exists(config.MODEL_PATH):
        os.makedirs(config.MODEL_PATH)
    criterion0 = LabelSmoothing(0.1)
    criterion1 = LabelSmoothing(0.1)
    criterion2 = LabelSmoothing(0.1)
    criterion3 = LabelSmoothing(0.1)
    criterion4 = LabelSmoothing(0.1)
    criterion5 = LabelSmoothing(0.1)
    criterion6 = LabelSmoothing(0.1)
    criterion7 = LabelSmoothing(0.1)
    criterion8 = LabelSmoothing(0.1)
    criterion9 = LabelSmoothing(0.1)
    criterion10 = LabelSmoothing(0.1)

    # ckpt_path = os.path.join('./exps/GCN/ckpt/', 'model_epoch_199.ckpt')
    # print('loading checkpoint from', ckpt_path)
    # checkpoint = torch.load(ckpt_path)
    # model.load_state_dict(checkpoint)
    params = list(model.parameters())
    # optimizer = torch.optim.Adam(params, lr=config.LEARNING_RATE)
    
    
    A_params =  [id(model.A)]
    base_params = filter(lambda p: id(p) not in A_params, model.parameters())
    optimizer = torch.optim.Adam([
                {'params': base_params},
                {'params': model.A, 'lr': config.LEARNING_RATE * 100}], lr=config.LEARNING_RATE)

    total_step = len(train_loader)
    min_val_loss = float('inf')

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=config.EPOCHES/3, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5,eta_min=4e-08)

    train_accs=[]
    train_loss =[]

    val_accs=[]
    val_loss =[]
    disease_nums = 11
    disease_names = ['主疾病','结节类型--高/等/低回声','结节类型--单/多发','边界','形态','纵横比','皮质类型','淋巴门结构','钙化','囊性区','血流']

    for epoch in range(config.EPOCHES):
        running_accs = [0. for _ in range(disease_nums)]
        print('TRAINING: Epoch [{}/{}]'.format(epoch, config.EPOCHES))
        for ii, (inputs,gt_labels) in enumerate(train_loader):

            for i in range(len(inputs)):
                # _input = _input.to(torch.float)
                inputs[i] = inputs[i].type(torch.FloatTensor)
                inputs[i] = inputs[i].to(device)

            # for _input in inputs:
            #     print(_input.device)

            for i in range(len(gt_labels)):
                gt_labels[i] = gt_labels[i].to(torch.long)
                gt_labels[i] = gt_labels[i].to(device)


            optimizer.zero_grad() 

            pred_outputs = model(inputs[0],inputs[1])

            loss0 = criterion0(pred_outputs[0],gt_labels[0])
            loss1 = criterion1(pred_outputs[1],gt_labels[1])
            loss2 = criterion2(pred_outputs[2],gt_labels[2])
            loss3 = criterion3(pred_outputs[3],gt_labels[3])
            loss4 = criterion4(pred_outputs[4],gt_labels[4])
            loss5 = criterion5(pred_outputs[5],gt_labels[5])
            loss6 = criterion6(pred_outputs[6],gt_labels[6])
            loss7 = criterion7(pred_outputs[7],gt_labels[7])
            loss8 = criterion8(pred_outputs[8],gt_labels[8])
            loss9 = criterion9(pred_outputs[9],gt_labels[9])
            loss10 = criterion10(pred_outputs[10],gt_labels[10])

            loss = loss0+loss1+5*loss2+loss3+5*loss4+5*loss5+5*loss6+5*loss7+loss8+loss9+5*loss10

 
            loss.backward()
            optimizer.step()
            

            for i in range(disease_nums):
                running_accs[i] += count_correct_nums(pred_outputs[i],gt_labels[i])

            if ii % config.LOG_STEP == 0:
                print(' Step [{}/{}], Loss: {:.4f}, '
                      .format(ii, total_step, loss.item(), ))
        scheduler.step()
        for i in range(len(running_accs)):
            running_accs[i] /= len(train_loader) * config.BATCH_SIZE

        #保存


        train_accs.append(running_accs)
        train_loss.append([l.item() for l in [loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10]])


    ################
    ###验证
    ################
        running_accs = [0. for _ in range(disease_nums)]
        with torch.no_grad():
            for ii, (inputs,gt_labels) in enumerate(val_loader):
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

                loss0 = criterion0(pred_outputs[0],gt_labels[0])
                loss1 = criterion1(pred_outputs[1],gt_labels[1])
                loss2 = criterion2(pred_outputs[2],gt_labels[2])
                loss3 = criterion3(pred_outputs[3],gt_labels[3])
                loss4 = criterion4(pred_outputs[4],gt_labels[4])
                loss5 = criterion5(pred_outputs[5],gt_labels[5])
                loss6 = criterion6(pred_outputs[6],gt_labels[6])
                loss7 = criterion7(pred_outputs[7],gt_labels[7])
                loss8 = criterion8(pred_outputs[8],gt_labels[8])
                loss9 = criterion9(pred_outputs[9],gt_labels[9])
                loss10 = criterion10(pred_outputs[10],gt_labels[10])

                loss = 10*loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10

                for i in range(disease_nums):
                    running_accs[i] += count_correct_nums(pred_outputs[i],gt_labels[i])

          

            for i in range(len(running_accs)):
                running_accs[i] /= len(train_loader) * config.BATCH_SIZE

        print('VALADATION: Epoch [{}/{}],loss:{}'.format(epoch, config.EPOCHES,loss.item()))
        val_accs.append(running_accs)
        val_loss.append([l.item() for l in [loss0,loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10]])


        if loss.item()<min_val_loss:
            torch.save(model.state_dict(), os.path.join(config.MODEL_PATH, 'model_epoch.ckpt'))

            print('saving model to ' + str(os.path.join(config.MODEL_PATH, 'model.ckpt')))


        plt.figure(figsize=(24, 8))


        temp_train_accs = list(  zip(*train_accs) )[::-1] #转置
        temp_val_accs = list(  zip(*val_accs) )[::-1] #转置
        for i in range(len(disease_names)):
            plt.plot(temp_train_accs[i], label='training acc_'+ str(i))
            plt.plot(temp_val_accs[i], label='validate acc_'+ str(i) )

        plt.legend(frameon=False)
        plt.savefig('acc.png')

        temp_train_loss = list(  zip(*train_loss) )[::-1] #转置
        temp_val_loss = list(  zip(*val_loss) )[::-1] #转置
        for i in range(len(disease_names)):
            plt.plot(temp_train_loss[i], label='training loss_'+ str(i) )
            plt.plot(temp_val_loss[i], label='validate loss_'+ str(i) )
        plt.legend(frameon=False)
        plt.savefig('loss.png')
        plt.clf()
        plt.close()
if __name__ == '__main__':
    main()


