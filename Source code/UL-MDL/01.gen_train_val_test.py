# -*- coding: utf-8 -*-
import os 
import random
us_dir = '../data/04.inpainted'
dpl_dir ='../data/03.registration'
label = 'img_label_list.txt'

random.seed(2020)

with open(label,'w',encoding='utf-8') as f:
	for img_name in os.listdir(us_dir):
		us_image_path = us_dir +'/'+ img_name
		dpl_image_path = dpl_dir +'/'+ img_name.replace('_0.bmp','_1.bmp').replace('_inpainted','')
		c = -1
		if ('fyx' in img_name):
			c = '反应性'
		elif ('Wzl' in img_name):
			c = '正常'	
		else:
			c = '恶性'
		print(img_name,str(c))
		f.write(us_image_path +'\t'+dpl_image_path+'\t'+c+'\n')

train = 0.6
val = 0.2
test = 1 - train - val

train_output = 'train.txt'
val_output = 'val.txt'
test_output = 'test.txt'


with open(label, 'r',encoding="utf-8") as lf:
    lines = list(lf.readlines())
    random.shuffle(lines)
    print("gen train file for %f of the total file count %d" %(train, len(lines)))
    with open(train_output, 'w',encoding="utf-8") as train_file:
        for filename in lines[0:round(len(lines) * train)]:
            # print(filename)
            train_file.write(filename)
    print("gen val file for %f of the total file count %d" %(val, len(lines)))
    with open(val_output, 'w',encoding="utf-8") as val_file:
        for filename in lines[round(len(lines) * train):round(len(lines) * (train + val))]:
            val_file.write(filename)
    print("gen test file for %f of the total file count %d" %(test, len(lines)))
    with open(test_output, 'w',encoding="utf-8") as test_file:
        for filename in lines[round(len(lines) * (train + val)): -1]:
            test_file.write(filename)

