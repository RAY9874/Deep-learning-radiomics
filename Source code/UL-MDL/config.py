# -*- coding: utf-8 -*-
import numpy as np
import os
# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):

    EXP_NAME = 'GCN'
    IMG_MAX_DIM= 256
    IMG_MIN_DIM = 200
    BATCH_SIZE = 6
    NORMALIZE = True
    AUGMENT= False
    EPOCHES = 300
    
    DISEASE_LABELS = ['正常','反应性','恶性']
    ROOT_PATH = '../'
    BINGLI = ROOT_PATH+'../data/淋巴超声报告_新指标_模型输入_dummy.xlsx'
    TRAIN_LIST = ROOT_PATH+'train.txt'
    VAL_LIST = ROOT_PATH+'val.txt'
    TEST_LIST = ROOT_PATH+'test.txt'
    LEARNING_RATE = 1e-3
    LOG_STEP = 20
    SAVE_STEP = 'Each epoch'
    MODEL_PATH = './exps/'+EXP_NAME+'/ckpt/'

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        # print('config initialize')
        self.display()

    def display(self):
        """Display Configuration values."""

        print("\n"+'-'*33+'Configurations'+'-'*33+'\n')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print('\n'+ '-' * 80)
