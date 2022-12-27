#套件
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

import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.utils.rnn as rnn_utils
import random
import torch.nn.functional as F


#分類網路 function

class Residual(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=2, stride=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv4res = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=1)
        self.AvgPool1d=nn.AdaptiveAvgPool1d(12)
        self.bn4res = nn.BatchNorm1d(out_channels)

    def forward(self, X):
        
        Y1 = self.conv4res(X)
        Y1 = self.bn4res(Y1)

        Y2 = self.bn1(self.conv1(X))

        Y2 = F.relu(Y2)

        Y2 = self.bn2(self.conv2(Y2))

        Y2 = F.relu(Y2)

        return Y1 + Y2

def resnet_block(in_channels, out_channels, num_residuals):

    blk = []
    for i in range(num_residuals):
        blk.append(Residual(in_channels, out_channels))

    return nn.Sequential(*blk)

class LSTM_FCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(LSTM_FCN, self).__init__()
        # LSTM
        self.conv4lstm = nn.Conv1d(48, 48, kernel_size=1, stride=1)
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)

        # 1D conv
        self.conv1 = nn.Conv1d(1, 256, kernel_size=2, stride=2)
        self.res_block=resnet_block(in_channels=256,out_channels=256,num_residuals=3)
        self.conv2 = nn.Conv1d(256, 16, kernel_size=2, stride=2)
        self.AvgPool1d=nn.AdaptiveAvgPool1d(48) # length

        # concat softmax
        self.fc1 = nn.Linear(768, 256, bias=True)
        self.dropou1=nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64, bias=False)
        self.dropou2=nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, output_dim, bias=False)
        self.softmax=nn.Softmax(dim=1)

    def init_model(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(m.weight)
        print("init success !!")

    def forward(self, x):
        
        
        
        # x=torch.unsqueeze(x,0)
        # 1D cnn
        cnn_out=self.conv1(x)
        cnn_out=self.res_block(cnn_out)
        cnn_out=self.conv2(cnn_out)
        cnn_out=self.AvgPool1d(cnn_out)            

        y=torch.flatten(cnn_out,start_dim=1)
   
        y = self.fc1(y) 
        y=self.dropou1(y)
  
        y = self.fc2(y) 
        y=self.dropou2(y)
        y = self.fc3(y) 

        y= self.softmax(y)
 
        return y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MyResModel = LSTM_FCN(input_dim=1, hidden_dim=16, output_dim=5, layers=1).to(device)
MyResModel.init_model()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(LSTM.parameters(),lr=0.001)
optimizer = torch.optim.SGD(MyResModel.parameters(), lr=0.002, momentum=0.9)



with open('pickle/cifar_train_tensor.pickle', 'rb') as f:
    train_tensor = pickle.load(f)
with open('pickle/cifar_train_label.pickle', 'rb') as f:
    train_label = pickle.load(f)
with open('pickle/cifar_val_tensor.pickle', 'rb') as f:
    val_tensor = pickle.load(f)
with open('pickle/cifar_val_label.pickle', 'rb') as f:
    val_label = pickle.load(f)

with open('pickle/cifar_train_class_label.pickle', 'rb') as f:
    train_class_label = pickle.load(f)

with open('pickle/cifar_val_class_label.pickle', 'rb') as f:
    val_class_label = pickle.load(f)
with open('pickle/cifar_class/cifar_class0_label.pickle', 'rb') as f:
    label_ = pickle.load(f)


#train分類模型

Epochs=500
max_test_acc=0.0
class_number = 10
train_loss_list=[]
train_acc_list=[]
test_loss_list=[]
test_acc_list=[]
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''按batch size包裝資料'''
register_1 = []
register_2 = []
train_tensor_list = []
train_label_list = []
count = 0
for t in range(len(train_tensor)):
    count+=1
    register_1.append(train_tensor[t])
    register_2.append(label_[t])
    if count % batch_size ==0:
        a = torch.stack(register_1).to(device)
        b = torch.stack(register_2)
        train_tensor_list.append(a)
        train_label_list.append(b)
        register_1 = []
        register_2 = []

for epoch in range(Epochs):
    train_loss=0.0
    train_acc=0.0
    test_loss=0.0
    test_acc=0.0
    MyResModel.train()
    count = 0
    for data_ in train_tensor_list:
        print(data_.shape)
        optimizer.zero_grad()
        pred = MyResModel(data_.to(device))
        print(pred.shape)
        print(torch.tensor(train_label_list[count]).shape)
        loss = criterion(pred.to(device),train_label_list[count].to(device))
        count+=1
        loss.backward()
        optimizer.step()
    MyResModel.eval()