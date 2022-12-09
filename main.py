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

img_dir = ''
save_dir = ''
train_val_num = 0.1
function.split_dataset(img_dir,save_dir,train_val_num,name1='10_percent',name2='90_percent') #切分出10%資料

percent10_dir = os.path.join(save_dir,'10_percent') 
train_val_num = 0.8
function.split_dataset(percent10_dir,percent10_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_val_num = 0.5
function.split_dataset(percent10_dir,save_dir,train_val_num,name1='first5_percent',name2='second5_percent') #將10%資料切成兩個5%資料

first5_dir = os.path.join(save_dir,'first5_percent')
train_val_num = 0.8
function.split_dataset(first5_dir,first5_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_path = (first5_dir+'_split')

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
    parser.add_argument("--class_num",type=int,default=100,help="class num") #類別數
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
    parser.add_argument("--class_num",type=int,default=100,help="class num") #類別數
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

