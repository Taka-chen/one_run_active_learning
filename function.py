#定義function

from torchvision import datasets,transforms
import random
import cv2
import glob
from pathlib import Path
import os
import torch
import torch.nn.functional as FUN
from torch.autograd import Variable
import torchvision.transforms as T

#把圖片填充成正方形，防止resize後變形
def expend_img(img):
    fill_pix=[0,0,0] #填充色素，可自己設定
    h,w=img.shape[:2]
    if h>=w: #左右填充
        padd_width=int(h-w)//2
        padd_top,padd_bottom,padd_left,padd_right=0,0,padd_width,padd_width #各個方向的填充像素
    elif h<w: #上下填充
        padd_high=int(w-h)//2
        padd_top,padd_bottom,padd_left,padd_right=padd_high,padd_high,0,0 #各個方向的填充像素
    new_img = cv2.copyMakeBorder(img,padd_top,padd_bottom,padd_left,padd_right,cv2.BORDER_CONSTANT, value=fill_pix)
    return new_img

#切分訓練集和測試集，並進行補邊處理
def split_dataset(img_dir,save_dir,train_val_num,name1,name2):
    '''
    :param img_dir: 原始图片路径，注意是所有类别所在文件夹的上一级目录
    :param save_dir: 保存图片路径
    :param train_val_num: 切分比例
    :return:
    '''
    img_dir_list=glob.glob(img_dir+os.sep+"*")#獲取每個類别所在的路徑（一個類别對應一個文件夾）
    for class_dir in img_dir_list:
        class_name=class_dir.split(os.sep)[-1] #獲取當前類别
        img_list=glob.glob(class_dir+os.sep+"*") #獲取每個類别文件夾下的所有圖片
        all_num=len(img_list) #獲取總個數
        train_list=random.sample(img_list,int(all_num*train_val_num)) #訓練集圖片所在路徑
        save_train=save_dir+os.sep+name1+os.sep+class_name
        save_val=save_dir+os.sep+name2+os.sep+class_name
        os.makedirs(save_train,exist_ok=True) #建立對應的文件夾
        os.makedirs(save_val,exist_ok=True) 
        #保存切分好的數據集
        for imgpath in img_list:
            imgname=Path(imgpath).name #獲取文件名
            if imgpath in train_list:
                img=cv2.imread(imgpath)
                new_img=expend_img(img)
                cv2.imwrite(save_train+os.sep+imgname,new_img)
            else: #將除了訓練集意外的數據均視為驗證集
                img = cv2.imread(imgpath)
                new_img = expend_img(img)
                cv2.imwrite(save_val + os.sep + imgname, new_img)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class ImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, *args):
        super(ImageFolderWithPaths, self).__init__(*args)
        self.trans = args[1]
    def __len__(self):
      return len(self.imgs)
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        return (img, label ,path)

def loaddata(data_dir, batch_size, set_name, shuffle,input_size,means,stds):
    data_transforms = {}
    data_transforms[set_name] = \
            transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

    image_datasets = {x: ImageFolderWithPaths(os.path.join(data_dir, x), data_transforms[x]) for x in [set_name]}
    # num_workers=0 if CPU else = 1
    dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=batch_size,shuffle=shuffle, num_workers=0) for x in [set_name]}
    data_set_sizes = len(image_datasets[set_name])
    return dataset_loaders, data_set_sizes

def test_model(model, data_dir, batch_size,set_name):
    tensor = []
    class_num = []
    path = []
    model.eval()
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name=set_name, shuffle=False)
    for data in dset_loaders[set_name]:
        inputs, labels, paths = data #path抓出被分類的圖片的原始路徑
        labels = labels.type(torch.LongTensor)
        # GPU
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        tensor.append(outputs.data)
        class_num.append(labels)
        path.append(paths)
    return tensor, class_num, path

def model_acc(model, criterion, data_dir, batch_size,set_name):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_number = loaddata(data_dir=data_dir, batch_size=batch_size, set_name = set_name,shuffle=False)
    for data in dset_loaders[set_name]:
        inputs, labels, paths = data #path抓出被分類的圖片的原始路徑
        labels = labels.type(torch.LongTensor)
        # GPU
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        if cont == 0:
            outPre = outputs.data.cpu()
            outLabel = labels.data.cpu()
        else:
            outPre = torch.cat((outPre, outputs.data.cpu()), 0)
            outLabel = torch.cat((outLabel, labels.data.cpu()), 0)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        cont += len(labels)
        acc = running_corrects/cont
    loss = running_loss / dset_number
    acc = running_corrects.double() / dset_number
    return dset_number,loss,acc
