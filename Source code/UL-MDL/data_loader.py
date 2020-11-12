# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
from PIL import Image
import skimage
from distutils.version import LooseVersion
import skimage.transform as trans
import cv2
import os
from random import randint
import imgaug as ia
import random
import imgaug.augmenters as iaa
import pandas as pd
import random

class LymphDataset(data.Dataset):
    def __init__(self, data_list,config):
        self.config = config
        self.img_max_dim = config.IMG_MAX_DIM
        self.img_min_dim = config.IMG_MIN_DIM
        self.batch_size = config.BATCH_SIZE
        self.normalize_flag = config.NORMALIZE
        self.augment_flag = config.AUGMENT
        self.data_list = data_list
        self.bingli = pd.read_excel(config.BINGLI)
        #必须强制转换为str，否则是obj，在查询时 == 判断会出错
        self.bingli[['编号']] = self.bingli[['编号']].astype(str)


        with open(self.data_list, "r") as fin:
            self.lines = list(fin.readlines())

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        item = self.lines[index]
        item_get = self.get(item)
        if item_get is not None:
            return item_get
        else:
            item =None
            while(item is None):
                randnum = random.randint(0,len(self.lines))
                item = self.get(self.lines[randnum])
            return item
    def get(self, item):

    #读取
        a = item.strip().split("\t")
        us_image_path = '../'+a[0]
        dpl_image_path = '../'+a[1]
        c = a[2]

        # print(us_image_path,dpl_image_path)
        us_image = cv2.imread(us_image_path)
        dpl_image = cv2.imread(dpl_image_path)
    #增强
        if self.augment_flag:
            us_image = self.random_aug(us_image)
            dpl_image = self.random_aug(dpl_image)
    # 归一化
        if self.normalize_flag:
            us_image = self.normalize(us_image)
            dpl_image = self.normalize(dpl_image)

    # resize
        us_image, _, scale, _, _ = self.resize_image(img=us_image, min_dim=self.img_min_dim, max_dim=self.img_max_dim)
        dpl_image, _, scale, _, _ = self.resize_image(img=dpl_image, min_dim=self.img_min_dim, max_dim=self.img_max_dim)
        # numpy image: H x W x C # torch image: C x H x W，需要转换
        us_image = np.transpose(us_image,(2, 0, 1))
        dpl_image = np.transpose(dpl_image,(2, 0, 1))
    #多标签
        name = us_image_path.split('/')[-1]

        name = '_'.join(name.split('_')[:-2])
        name = name.replace('_a','a').replace('_b','b')
        # print(name)
        #主
        disease_label= self.config.DISEASE_LABELS.index(c)
        #数值型
        # changjing = self.get_changjing(name,self.bingli,max_changjing) * scale
        # duanjing = self.get_duanjing(name,self.bingli,max_duanjing) * scale
        # pizhihoudu = self.get_pizhihoudu(name,self.bingli,max_pizhihoudu) * scale
        # linbamenhengjing = self.get_linbamenhengjing(name,self.bingli,max_linbamenhengjing) * scale
        #类别型
        jiejieleixing_huisheng = np.argmax(self.get_jiejieleixing_huisheng(name,self.bingli))
        # jiejieleixing_fa = np.argmax(self.get_jiejieleixing_fa(name, self.bingli))
        bianjie = np.argmax(self.get_bianjie(name, self.bingli))
        xingtai = np.argmax(self.get_xingtai(name, self.bingli))
        zonghengbi = np.argmax(self.get_zonghengbi(name, self.bingli))
        pizhileixing = np.argmax(self.get_pizhileixing(name, self.bingli))
        linbamenjiegou = np.argmax(self.get_linbamenjiegou(name, self.bingli))
        gaihua = np.argmax(self.get_gaihua(name, self.bingli))
        nangxingqu = np.argmax(self.get_nangxingqu(name, self.bingli))
        xueliu = np.argmax(self.get_xueliu(name, self.bingli))

        inputs = [us_image,dpl_image]
        outputs = [disease_label,jiejieleixing_huisheng,
                    bianjie,xingtai,zonghengbi,pizhileixing,
                    linbamenjiegou,gaihua,nangxingqu,xueliu]
        return inputs,outputs

    def resize_image(self,img, min_dim=None, max_dim=None, min_scale=None, mode="square"):
        image = img
        image_dtype = image.dtype
        # Default window (y1, x1, y2, x2) and default scale == 1.
        h, w = image.shape[:2]
        window = (0, 0, h, w)
        scale = 1
        padding = [(0, 0), (0, 0), (0, 0)]
        crop = None
        if mode == "none":
            return image, window, scale, padding, crop

        # Scale?
        if min_dim:
            # Scale up but not down
            scale = max(1, min_dim / min(h, w))
        if min_scale and scale < min_scale:
            scale = min_scale

        # Does it exceed max dim?
        if max_dim and mode == "square":
            image_max = max(h, w)
            if round(image_max * scale) > max_dim:
                scale = max_dim / image_max

        # Resize image using bilinear interpolation
        if scale != 1:
            image = self.resize(image, (round(h * scale), round(w * scale)),
                           preserve_range=True)

        # Need padding or cropping?
        if mode == "square":
            # Get new height and width
            h, w = image.shape[:2]
            top_pad = (max_dim - h) // 2
            bottom_pad = max_dim - h - top_pad
            left_pad = (max_dim - w) // 2
            right_pad = max_dim - w - left_pad
            top_pad=int(top_pad)
            bottom_pad=int(bottom_pad)
            left_pad=int(left_pad)
            right_pad=int(right_pad)
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0.)

            window = (top_pad, left_pad, h + top_pad, w + left_pad)

        elif mode == "pad64":
            h, w = image.shape[:2]
            # Both sides must be divisible by 64
            assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
            # Height
            if h % 64 > 0:
                max_h = h - (h % 64) + 64
                top_pad = (max_h - h) // 2
                bottom_pad = max_h - h - top_pad
            else:
                top_pad = bottom_pad = 0
            # Width
            if w % 64 > 0:
                max_w = w - (w % 64) + 64
                left_pad = (max_w - w) // 2
                right_pad = max_w - w - left_pad
            else:
                left_pad = right_pad = 0
            padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
            image = np.pad(image, padding, mode='constant', constant_values=0)
            window = (top_pad, left_pad, h + top_pad, w + left_pad)
        elif mode == "crop":
            # Pick a random crop
            h, w = image.shape[:2]
            y = random.randint(0, (h - min_dim))
            x = random.randint(0, (w - min_dim))
            crop = (y, x, min_dim, min_dim)
            image = image[y:y + min_dim, x:x + min_dim]
            window = (0, 0, min_dim, min_dim)
        else:            raise Exception("Mode {} not supported".format(mode))
        return image.astype(image_dtype), window, scale, padding, crop

    def resize(self,image, output_shape, order=1, mode='constant', cval=0, clip=True,
               preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
        """A wrapper for Scikit-Image resize().

        Scikit-Image generates warnings on every call to resize() if it doesn't
        receive the right parameters. The right parameters depend on the version
        of skimage. This solves the problem by using different parameters per
        version. And it provides a central place to control resizing defaults.
        """
        if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
            # New in 0.14: anti_aliasing. Default it to False for backward
            # compatibility with skimage 0.13.
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range, anti_aliasing=anti_aliasing,
                anti_aliasing_sigma=anti_aliasing_sigma)
        else:
            return skimage.transform.resize(
                image, output_shape,
                order=order, mode=mode, cval=cval, clip=clip,
                preserve_range=preserve_range)


    def normalize(self,image):
        image = image.astype(np.float64)
        mean = np.mean(image)
        if np.max(image)!= 0 :
            image = image - np.min(image)
            image = image / np.max(image)
        return image
    def random_aug(self,us_img):
        seq1 = iaa.Sequential([
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        ])
        # iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        num1 = random.uniform(-0.1, 0.1)
        num1 = round(num1, 2)
        num2 = random.uniform(-0.1, 0.1)
        num2 = round(num2, 2)
        num3 = random.uniform(-0.1, 0.1)
        num3 = round(num3, 2)
        num4 = random.uniform(-0.1, 0.1)
        num4 = round(num4, 2)
        seq2 = iaa.Sequential([
                iaa.CropAndPad(percent=(num1, num2, num3, num4),
                    pad_cval=0)])
        seq3 = iaa.Sequential([
            # iaa.GaussianBlur(sigma=(0, 1.0)),  # blur images with a sigma of 0 to 3.0
            iaa.AllChannelsHistogramEqualization()
        ])
        seq4 = iaa.Sequential([
            iaa.Noop()
            # iaa.AllChannelsHistogramEqualization()
        ])

        seq5 = iaa.Sequential([iaa.Affine(
            rotate=(-45, 45), # rotate by -45 to +45 degrees
        )])

        seq6 = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)])

        seq_list = [seq1,seq4,seq5]
        seqs = random.sample(seq_list,1)
        for seq in seqs:
            us_img = seq.augment_image(us_img)
        return us_img
    #####
    ## 新版超声报告所用多输出
    #####

    def get_changjing(self, name, df, max_changjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['长径（px）']]
        changjing = temp_df.values[0]
        changjing = changjing / max_changjing
        return changjing

    def get_duanjing(self, name, df, max_duanjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['短径（px）']]
        duanjing = temp_df.values[0]
        duanjing = duanjing / max_duanjing
        return duanjing

    def get_pizhihoudu(self, name, df, max_pizhihoudu):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['皮质厚度']]
        pizhihoudu = temp_df.values[0]
        pizhihoudu = pizhihoudu / max_pizhihoudu
        return pizhihoudu

    def get_linbamenhengjing(self, name, df, max_linbamenhengjing):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['淋巴门横径']]
        linbamenhengjing = temp_df.values[0]
        linbamenhengjing = linbamenhengjing / max_linbamenhengjing
        return linbamenhengjing

    def get_jiejieleixing_huisheng(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['结节类型（高回声/低回声/等回声）_低回声',
                           '结节类型（高回声/低回声/等回声）_等回声',
                           '结节类型（高回声/低回声/等回声）_高回声', ]]
        jiejieleixing_huisheng = temp_df.values[0]
        return jiejieleixing_huisheng

    def get_jiejieleixing_fa(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['结节类型（多发/单发）_单发',
                           '结节类型（多发/单发）_多发', ]]
        jiejieleixing_fa = temp_df.values[0]
        return jiejieleixing_fa

    def get_bianjie(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['边界（清/不）_不清',
                           '边界（清/不）_清']]
        bianjie = temp_df.values[0]
        return bianjie

    def get_xingtai(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['形态（规/不）_不规则',
                           '形态（规/不）_规则']]
        xingtai = temp_df.values[0]
        return xingtai

    def get_zonghengbi(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['纵横比＞2（有：>2/无：<2）(=2改为<2， 因：均为恶性)_<2',
                           '纵横比＞2（有：>2/无：<2）(=2改为<2， 因：均为恶性)_>2', ]]
        zonghengbi = temp_df.values[0]
        return zonghengbi

    def get_pizhileixing(self, name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['皮质类型（偏心型，不偏心型、无淋巴门）_不偏心',
                           '皮质类型（偏心型，不偏心型、无淋巴门）_偏心',
                           '皮质类型（偏心型，不偏心型、无淋巴门）_无淋巴门', ]]
        pizhileixing = temp_df.values[0]
        return pizhileixing

    def get_linbamenjiegou(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['淋巴门结构（偏心；有）_无淋巴门',
                           '淋巴门结构（偏心；有）_有', ]]
        linbamenjiegou = temp_df.values[0]
        return linbamenjiegou

    def get_gaihua(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['钙化（点状；粗大；无）_无钙化',
                           '钙化（点状；粗大；无）_点状钙化',
                           '钙化（点状；粗大；无）_粗大钙化']]
        gaihua = temp_df.values[0]
        return gaihua

    def get_nangxingqu(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[['囊性区（有；无）_无',
                           '囊性区（有；无）_有']]
        nangxingqu = temp_df.values[0]
        return nangxingqu

    def get_xueliu(self,name, df):
        temp_df = df[df['编号'] == name]
        temp_df = temp_df[
            ['血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_I',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_II',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_III',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_IV',
             '血流（Ⅰ型, 门型血流：居中的门样血流并有规则的放射状分支;Ⅱ型, 中心型血流：偏中心的门样血流伴或不伴不规则的放射状分支;Ⅲ型, 周围型血流：包膜下环绕血流 ;Ⅳ型, 混合型血流；V型：无血流）_V']]
        xueliu = temp_df.values[0]
        return xueliu


def get_loader(config,data_list, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    lymph = LymphDataset(data_list,config)
    print('Dataset inited , lenth :',len(lymph))
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=lymph,batch_size=config.BATCH_SIZE,
                                              shuffle=True, num_workers=num_workers)
    return data_loader

def my_collac_fn(batch):
    for i in range(len(batch)):
        if batch.data[i] == None:
            batch.data[i] = random.choice(batch.data)


if __name__ == '__main__':

    from config import Config
    config = Config()
    train_loader = get_loader(config,config.TEST_LIST, 1)
    for i, (inputs,gt_labels) in enumerate(train_loader):
        print(i)