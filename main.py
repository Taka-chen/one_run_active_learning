#用到的套件

from __future__ import print_function, division
import os  
import pandas as pd
import requests
import numpy as np
import re
import math
import torchvision
import cv2
import glob
import random
from pathlib import Path
from torchvision import datasets,transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import time
import argparse
from time import sleep
from tqdm import tqdm, trange
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as FUN
from scipy import io
import efficientnet_pytorch
import torchvision.transforms as T
import PIL
import pickle
import torchvision.datasets as dsets
from scipy.misc import imsave
import torch.nn.functional as F
import function
import main_model

device="cuda" if torch.cuda.is_available() else "cpu" 
class_num = 100

img_dir = ''
save_dir = ''
train_val_num = 0.1
function.split_dataset(img_dir,save_dir,train_val_num,name1='10_percent',name2='90_percent') #切分出10%資料
percent90_dir = os.path.join(save_dir,'90_percent')
percent10_dir = os.path.join(save_dir,'10_percent') 
train_val_num = 0.8
if not os.path.isdir(percent10_dir+'_split'):
    os.mkdir(percent10_dir+'_split')
function.split_dataset(percent10_dir,percent10_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_val_num = 0.5
function.split_dataset(percent10_dir,save_dir,train_val_num,name1='first5_percent',name2='second5_percent') #將10%資料切成兩個5%資料
second5_dir = os.path.join(save_dir,'second5_percent')
first5_dir = os.path.join(save_dir,'first5_percent')
train_val_num = 0.8
if not os.path.isdir(first5_dir+'_split'):
    os.mkdir(first5_dir+'_split')
function.split_dataset(first5_dir,first5_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_path = (first5_dir+'_split')
function.split_dataset(second5_dir,second5_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集

#train 5% data
#main_model参數設置
save_model_path = './weight'
first5_model_name = 'efficientb5_first5.pth'
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default="./model/efficientnet-b5-b6417697.pth",help='initial weights path')#預訓練模型路徑
    parser.add_argument("--img-dir",type=str,default=train_path,help="train image path") #數據集的路徑
    parser.add_argument("--imgsz",type=int,default=224,help="image size") #圖像尺寸
    parser.add_argument("--epochs",type=int,default=50,help="train epochs")#訓練批次
    parser.add_argument("--batch-size",type=int,default=8,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=class_num,help="class num") #類別數
    parser.add_argument("--lr",type=float,default=0.0005,help="Init lr") #學習率初始值
    parser.add_argument("--m",type=float,default=0.9,help="optimer momentum") #動量
    parser.add_argument("--save-dir" , type=str , default=save_model_path , help="save models dir")#保存模型路徑
    parser.add_argument("--model_name" , type=str , default="efficientnet-b5" , help="model version")#選用的efficientnet版本
    parser.add_argument("--save-model-name" , type=str , default=first5_model_name , help="save models name")#保存模型檔名
    opt=parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt=parse_opt()
    models=main_model.Efficientnet_train(opt)
    models()

#train 10% data
#main_model参數設置
train_path = (percent10_dir+'_split')
pretrain_weight_path = os.path.join(save_model_path,first5_model_name)
percent10_model_name = 'efficientb5_10percent.pth'
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights" , type=str,default=pretrain_weight_path , help='initial weights path')#預訓練模型路徑
    parser.add_argument("--img-dir",type=str,default=train_path,help="train image path") #數據集的路徑
    parser.add_argument("--imgsz",type=int,default=224,help="image size") #圖像尺寸
    parser.add_argument("--epochs",type=int,default=50,help="train epochs")#訓練批次
    parser.add_argument("--batch-size",type=int,default=8,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=class_num,help="class num") #類別數
    parser.add_argument("--lr",type=float,default=0.0005,help="Init lr") #學習率初始值
    parser.add_argument("--m",type=float,default=0.9,help="optimer momentum") #動量
    parser.add_argument("--save-dir" , type=str , default=save_model_path , help="save models dir")#保存模型路徑
    parser.add_argument("--model_name" , type=str , default="efficientnet-b5" , help="model version")#選用的efficientnet版本
    parser.add_argument("--save-model-name" , type=str , default=percent10_model_name , help="save models name")#保存模型檔名
    opt=parser.parse_known_args()[0]
    return opt

if __name__ == '__main__':
    opt=parse_opt()
    models=main_model.Efficientnet_train(opt)
    models()



#產出對second 5 data的confidence
input_size = 224
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
unorm = function.UnNormalize(mean = means, std = stds)
device="cuda" if torch.cuda.is_available() else "cpu" 
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net_name = 'efficientnet-b5'
    data_dir = second5_dir+'_split'
    weight_dir = './weight'
    set_list = os.listdir(data_dir)
    if not os.path.isdir('pickle'):
        os.mkdir('pickle')
    pickle_dir = './pickle'
    for name in set_list:
        #以 first 5% 模型產出對second 5 data的confidence
        modelft_file = os.path.join(weight_dir,first5_model_name)
        batch_size = 1
        model = efficientnet_pytorch.EfficientNet.from_name(net_name) #有GPU時用
        # 修改全連接層
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, class_num)
        model = model.to(device)
        #load model
        model.load_state_dict(torch.load(modelft_file))
        criterion = nn.CrossEntropyLoss().cuda()
        os.makedirs(pickle_dir,exist_ok=True)
        first5_confidence_pickle_dir = os.path.join(pickle_dir,name+'_first5_confidence.pickle')
        first5_classnum_pickle_dir = os.path.join(pickle_dir,name+'_first5_classnum.pickle')
        first5_path_pickle_dir = os.path.join(pickle_dir,name+'_first5_path.pickle')
        with open(first5_confidence_pickle_dir , 'wb') as f:
            pickle.dump(tensor, f)
        with open(first5_classnum_pickle_dir, 'wb') as f:
            pickle.dump(class_num, f)
        with open(first5_path_pickle_dir, 'wb') as f:
            pickle.dump(path, f)
        #以 10% 模型產出對second 5 data的confidence
        modelft_file = os.path.join(weight_dir,percent10_model_name)
        model = efficientnet_pytorch.EfficientNet.from_name(net_name) #有GPU時用
        # 修改全連接層
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, class_num)
        model = model.to(device)
        #load model
        model.load_state_dict(torch.load(modelft_file))
        criterion = nn.CrossEntropyLoss().cuda()
        tensor, class_num, path = function.test_model(model ,data_dir,batch_size,set_name = '')
        os.makedirs(pickle_dir,exist_ok=True)
        percent10_confidence_pickle_dir = os.path.join(pickle_dir,name+'_percent10_confidence.pickle')
        percent10_classnum_pickle_dir = os.path.join(pickle_dir,name+'_percent10_classnum.pickle')
        percent10_path_pickle_dir = os.path.join(pickle_dir,name+'_percent10_path.pickle')
        with open(percent10_confidence_pickle_dir , 'wb') as f:
            pickle.dump(tensor, f)
        with open(percent10_classnum_pickle_dir, 'wb') as f:
            pickle.dump(class_num, f)
        with open(percent10_path_pickle_dir, 'wb') as f:
            pickle.dump(path, f)

#製作min model要用的label和data( 只取最大的x個值 )
#label類別 0:上升 1:不變 2:下降
with open(os.path.join(pickle_dir,'train_first5_confidence.pickle'), 'rb') as f:
    train_data = pickle.load(f)
with open(os.path.join(pickle_dir,'train_percent10_confidence.pickle'), 'rb') as f:
    train_label = pickle.load(f)
with open(os.path.join(pickle_dir,'val_first5_confidence.pickle'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(pickle_dir,'val_percent10_confidence.pickle'), 'rb') as f:
    val_label = pickle.load(f)
for class_ in range(class_num):
    t_label = []
    t_data = []
    v_label = []
    v_data = []
    save_pickle_dir = os.path.join(pickle_dir,'min_model',str(class_))
    os.makedirs(save_pickle_dir, exist_ok=True)
    for t in range(len(train_data)):
        value , index = torch.sort(train_data[t].squeeze(),descending = True)
        for t2 in range(int(len(index)/20)):
            if int(index[t2]) == class_:
                t_data.append(train_data[t])
                differ = train_label[t][0][class_] - train_data[t][0][class_]
                if differ >= 0:
                    increase = abs(differ) / abs(train_data[t][0][class_])
                    if increase > 1 :
                        classifi = 0
                    elif increase<=1:
                        classifi = 1
                else:
                    reduce = abs(differ) / abs(train_data[t][0][class_])
                    if reduce > 0.5 :
                        classifi = 2
                    elif reduce<=0.5:
                        classifi = 1
                t_label.append(torch.tensor(classifi))
    data = torch.stack(t_data)
    label = torch.stack(t_label)
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_train_data.pickle'), 'wb') as f:
        pickle.dump(data, f)
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_train_label.pickle'), 'wb') as f:
        pickle.dump(label, f)
    for t in range(len(val_data)):
        value , index = torch.sort(val_data[t].squeeze(),descending = True)
        # for t2 in range(int(len(index)/33)):
        for t2 in range(int(len(index)/20)):

            if int(index[t2]) == class_:
                v_data.append(val_data[t])
                differ = val_label[t][0][class_] - val_data[t][0][class_]
                if differ >= 0:
                    increase = abs(differ) / abs(val_data[t][0][class_])
                    if increase > 1 :
                        classifi = 0
                    elif increase<=1:
                        classifi = 1
                else:
                    reduce = abs(differ) / abs(val_data[t][0][class_])
                    if reduce > 0.5 :
                        classifi = 2
                    elif reduce<=0.5:
                        classifi = 1
                v_label.append(torch.tensor(classifi))
    data2 = torch.stack(v_data)
    label2 = torch.stack(v_label)
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_val_data.pickle'), 'wb') as f:
        pickle.dump(data2, f)
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_val_label.pickle'), 'wb') as f:
        pickle.dump(label2, f)
