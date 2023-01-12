#main_model training

import function
from torchvision import datasets,transforms
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm, trange
import time
import os  
import torch.nn.functional as FUN
from torch.autograd import Variable
import torchvision.transforms as T

device="cuda" if torch.cuda.is_available() else "cpu" 

class Efficientnet_train():
    def __init__(self,opt):
        self.epochs=opt.epochs #訓練週期
        self.batch_size=opt.batch_size #batch_size
        self.class_num=opt.class_num #類别數
        self.imgsz=opt.imgsz #圖片尺寸
        self.img_dir=opt.img_dir #圖片路徑
        self.weights=opt.weights #模型路徑
        self.save_dir=opt.save_dir #保存模型路徑
        self.save_model_name=opt.save_model_name #保存模型檔名
        self.lr=opt.lr #初始化學習率
        self.moment=opt.m #動量
        self.version = opt.model_name
        base_model = EfficientNet.from_name(self.version) #加載模型，使用b幾的就改為b幾
        state_dict = torch.load(self.weights)
        base_model.load_state_dict(state_dict)
        # 修改全連接層
        num_ftrs = base_model._fc.in_features
        base_model._fc = nn.Linear(num_ftrs, self.class_num)
        self.model = base_model.to(device)
        # 交叉熵損失函數
        self.cross = nn.CrossEntropyLoss()
        # 優化器
        self.optimzer = optim.SGD((self.model.parameters()), lr=self.lr, momentum=self.moment, weight_decay=0.0004)

        #獲取處理後的數據集和類别映射表
        self.trainx,self.valx,self.b=self.process()
        print(self.b)
    def __call__(self):
        best_acc = 0
        self.model.train(True)
        for ech in tqdm(range(self.epochs)):
            optimzer1 = self.lrfn(ech, self.optimzer)

            print("----------Start Train Epoch %d----------" % (ech + 1))
            # 開始訓練
            run_loss = 0.0  # 損失
            run_correct = 0.0  # 準確率
            count = 0.0  # 分類正確的個數

            for i, data in enumerate(self.trainx):
                # print('train')
                inputs, label = data
                inputs, label = inputs.to(device), label.to(device)

                # 訓練
                optimzer1.zero_grad()
                output = self.model(inputs)

                loss = self.cross(output, label)
                loss.backward()
                optimzer1.step()

                run_loss += loss.item()  # 損失累加
                _, pred = torch.max(output.data, 1)
                count += label.size(0)  # 求總共的訓練個數
                run_correct += pred.eq(label.data).cpu().sum()  # 截止當前預測正確的個數
                #每隔100個batch顯示一次信息，這裡顯示的ACC是當前預測正確的個數/當前訓練過的個數
                if (i+1)%100==0:
                    print('[Epoch:{}__iter:{}/{}] | Acc:{}'.format(ech + 1,i+1,len(self.trainx), run_correct/count))
            train_acc = run_correct / count
            # 每次訓完一批顯示一次信息
            print('Epoch:{} | Loss:{} | Acc:{}'.format(ech + 1, run_loss / len(self.trainx), train_acc))

            # 訓完一批次後進行驗證
            print("----------Waiting Test Epoch {}----------".format(ech + 1))
            with torch.no_grad():
                correct = 0.  # 預測正確的個數
                total = 0.  # 總個數
                for inputs, labels in self.valx:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)

                    # 穫取最高分的那個類的索引
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += pred.eq(labels).cpu().sum()
                test_acc = correct / total
                print("批次%d的验证集准确率" % (ech + 1), correct / total)
            if best_acc < test_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir,self.save_model_name) )#不加state_dict()存法會直接把模型架構和權重一起存入weight檔中
                                                                                                       #加state_dict()則只單純存權重(不易報錯)

  #數據處理
    def process(self):
        # 數據增强
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.RandomRotation(45),  # 随機旋轉，旋轉範圍為【-45,45】
                transforms.ToTensor(),  # 轉換為張量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
            ]),
            "val": transforms.Compose([
                transforms.Resize((self.imgsz, self.imgsz)),  # resize
                transforms.CenterCrop((self.imgsz, self.imgsz)),  # 中心裁剪
                transforms.ToTensor(),  # 張量轉換
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

        # 定義圖像生成器
        image_datasets = {x: datasets.ImageFolder(root=os.path.join(self.img_dir,x), transform=data_transforms[x]) for x in ['train', 'val']}

        # 得到訓練集和驗證集
        trainx = DataLoader(image_datasets["train"], batch_size=self.batch_size, shuffle=True, drop_last=True)
        valx = DataLoader(image_datasets["val"], batch_size=self.batch_size, shuffle=True, drop_last=False)
        b = image_datasets["train"].class_to_idx  # id和類别對
        return trainx,valx,b

    # 動態調整學習率
    def lrfn(self,num_epoch, optimzer):
        lr_start = 0.00001  # 初始值
        max_lr = 0.0004  # 最大值
        lr_up_epoch = 15  # 學習率上升的epoch數
        lr_sustain_epoch = 5  # 學習率保持不變的epoch數
        lr_exp = .8  # 學習率衰减因子
        if num_epoch < lr_up_epoch:  # 學習率線性增加
            lr = (max_lr - lr_start) / lr_up_epoch * num_epoch + lr_start
        elif num_epoch < lr_up_epoch + lr_sustain_epoch:  # 學習率保持不變
            lr = max_lr
        else:  # 指數下降
            lr = (max_lr - lr_start) * lr_exp ** (num_epoch - lr_up_epoch - lr_sustain_epoch) + lr_start
        for param_group in optimzer.param_groups:
            param_group['lr'] = lr
        return optimzer


tensor = []
label = []
class_num = []
path = []
input_size = 224
#把要進行test的data反向正規化成原圖
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
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
unorm = UnNormalize(mean = means, std = stds)

#載入測試集
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

#產出對目標的confidence
def test_model(model, data_dir, batch_size,set_name):
    tensor = []
    class_num = []
    path = []
    model.eval()
    dset_loaders, dset_sizes = loaddata(data_dir=data_dir, batch_size=batch_size, set_name=set_name, shuffle=False, input_size=input_size, means=means, stds=stds)
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

#產出對某一資料的準確率
def model_acc(model, criterion, data_dir, batch_size,set_name):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    cont = 0
    outPre = []
    outLabel = []
    dset_loaders, dset_number = loaddata(data_dir=data_dir, batch_size=batch_size, set_name = set_name,shuffle=False, input_size=input_size, means=means, stds=stds)
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

#min model分類網路

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
        Y2 = FUN.relu(Y2)
        Y2 = self.bn2(self.conv2(Y2))
        Y2 = FUN.relu(Y2)
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
        self.conv4lstm = nn.Conv1d(48, 48, kernel_size=2, stride=2)
        self.rnn = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        # 1D conv
        self.conv1 = nn.Conv1d(1, 256, kernel_size=2, stride=2)
        self.res_block=resnet_block(in_channels=256,out_channels=256,num_residuals=3)
        self.conv2 = nn.Conv1d(256, 16, kernel_size=1, stride=1)
        self.AvgPool1d=nn.AdaptiveAvgPool1d(48) # length
        # concat softmax
        self.fc1 = nn.Linear(768, 256, bias=True)
        self.dropou1=nn.Dropout(0.7)
        self.fc2 = nn.Linear(256, 64, bias=False)
        self.dropou2=nn.Dropout(0.7)
        self.fc3 = nn.Linear(64, output_dim, bias=False)
        self.softmax=nn.Softmax(dim=1)
    def init_model(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d)):
                torch.nn.init.xavier_uniform_(m.weight)
        print("init success !!")
    def forward(self, x):  # x=torch.unsqueeze(x,0)   
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

