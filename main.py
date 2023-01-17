#用到的套件

from __future__ import print_function, division
import os  
import re
import cv2
import glob
import torch
import torch.nn as nn
import time
import argparse
from torch.autograd import Variable
import torch.nn.functional as FUN
import efficientnet_pytorch
import pickle
import function
import main_model

start_time = time.perf_counter() #程式初始時間

device="cuda" if torch.cuda.is_available() else "cpu" 
class_number = 10

img_dir = '../all_file/data/cifar10'
save_dir = '../all_file/data'
train_val_num = 0.1
if not os.path.isdir(os.path.join(save_dir,'10_percent')) and not os.path.isdir(os.path.join(save_dir,'90_percent')):
    function.split_dataset(img_dir,save_dir,train_val_num,name1='10_percent',name2='90_percent') #切分出10%資料
percent90_dir = os.path.join(save_dir,'90_percent')
percent10_dir = os.path.join(save_dir,'10_percent') 
train_val_num = 0.8
os.makedirs(percent10_dir+'_split',exist_ok=True)
if not os.path.isdir(os.path.join(percent10_dir+'_split','train')) and not os.path.isdir(os.path.join(percent10_dir+'_split','val')):
    function.split_dataset(percent10_dir,percent10_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_val_num = 0.5
if not os.path.isdir(os.path.join(save_dir,'first5_percent')) and not os.path.isdir(os.path.join(save_dir,'second5_percent')):  
    function.split_dataset(percent10_dir,save_dir,train_val_num,name1='first5_percent',name2='second5_percent') #將10%資料切成兩個5%資料
second5_dir = os.path.join(save_dir,'second5_percent')
first5_dir = os.path.join(save_dir,'first5_percent')
train_val_num = 0.8
os.makedirs(first5_dir+'_split',exist_ok=True)
if not os.path.isdir(os.path.join(first5_dir+'_split','train')) and not os.path.isdir(os.path.join(first5_dir+'_split','val')):
    function.split_dataset(first5_dir,first5_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集
train_path = (first5_dir+'_split')
if not os.path.isdir(os.path.join(second5_dir+'_split','train')) and not os.path.isdir(os.path.join(second5_dir+'_split','val')):
    function.split_dataset(second5_dir,second5_dir+'_split',train_val_num,name1='train',name2='val') #將5%資料切分出訓練集和測試集

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("Data preparation completed time: ", "{:.0f}".format(elapsed_time))

#///train 5% data///
#main_model参數設置
save_model_path = '../all_file/weight'
first5_model_name = 'efficientb5_first5.pth'
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights",type=str,default="../all_file/model/efficientnet-b5-b6417697.pth",help='initial weights path')#預訓練模型路徑
    parser.add_argument("--img-dir",type=str,default=train_path,help="train image path") #數據集的路徑
    parser.add_argument("--imgsz",type=int,default=224,help="image size") #圖像尺寸
    parser.add_argument("--epochs",type=int,default=1,help="train epochs")#訓練批次
    parser.add_argument("--batch-size",type=int,default=8,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=class_number,help="class num") #類別數
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

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("5% model training completed time: ", "{:.0f}".format(elapsed_time))

#///train 10% data///
#main_model参數設置
train_path = (percent10_dir+'_split')
percent10_model_name = 'efficientb5_10percent.pth'
def parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument("--weights" , type=str,default="../all_file/model/efficientnet-b5-b6417697.pth" , help='initial weights path')#預訓練模型路徑
    parser.add_argument("--img-dir",type=str,default=train_path,help="train image path") #數據集的路徑
    parser.add_argument("--imgsz",type=int,default=224,help="image size") #圖像尺寸
    parser.add_argument("--epochs",type=int,default=1,help="train epochs")#訓練批次
    parser.add_argument("--batch-size",type=int,default=8,help="train batch-size") #batch-size
    parser.add_argument("--class_num",type=int,default=class_number,help="class num") #類別數
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

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("10% model training completed time: ", "{:.0f}".format(elapsed_time))

#///產出對second 5 data的confidence///
input_size = 224
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
unorm = main_model.UnNormalize(mean = means, std = stds)
device="cuda" if torch.cuda.is_available() else "cpu" 
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net_name = 'efficientnet-b5'
    data_dir = second5_dir+'_split'
    weight_dir = '../all_file/weight'
    set_list = os.listdir(data_dir)
    if not os.path.isdir('pickle'):
        os.mkdir('pickle')
    pickle_dir = '../all_file/pickle'
  
    #///定義 first5 model///
    modelft_file = os.path.join(weight_dir,first5_model_name)
    batch_size = 1
    # GPU時
    model = efficientnet_pytorch.EfficientNet.from_name(net_name)
    # 修改全連接層
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, class_number)
    model = model.to(device)
    #load model
    model.load_state_dict(torch.load(modelft_file))
    criterion = nn.CrossEntropyLoss().cuda()
    set_list = os.listdir(data_dir)

    #///定義 10% model///
    model2 = efficientnet_pytorch.EfficientNet.from_name(net_name) #有GPU時用
    # 修改全連接層
    num_ftrs2 = model2._fc.in_features
    model2._fc = nn.Linear(num_ftrs2, class_number)
    model2 = model2.to(device)
    #load model
    model2.load_state_dict(torch.load(modelft_file))

    for name in set_list:
        #///使用 first5 model///
        tensor, pre_result, path = main_model.test_model(model ,data_dir,batch_size,set_name = name)
        pickle_dir = '../all_file/pickle'
        os.makedirs(pickle_dir,exist_ok=True)
        first5_confidence_pickle_dir = os.path.join(pickle_dir,'first5M_secondD_'+name+'_confidence.pickle')
        first5_classnum_pickle_dir = os.path.join(pickle_dir,'first5M_secondD_'+name+'_classnum.pickle')
        first5_path_pickle_dir = os.path.join(pickle_dir,'first5M_secondD_'+name+'_path.pickle')
        with open(first5_confidence_pickle_dir , 'wb') as f:
            pickle.dump(tensor, f)
        with open(first5_classnum_pickle_dir, 'wb') as f:
            pickle.dump(pre_result, f)
        with open(first5_path_pickle_dir, 'wb') as f:
            pickle.dump(path, f)
        #///使用 10% model///
        tensor2, pre_result2, path2 = main_model.test_model(model2 ,data_dir,batch_size,set_name = name)
        pickle_dir2 = '../all_file/pickle'
        os.makedirs(pickle_dir2,exist_ok=True)
        percent10_confidence_pickle_dir = os.path.join(pickle_dir2,'percent10M_secondD_'+name+'_confidence.pickle')
        percent10_classnum_pickle_dir = os.path.join(pickle_dir2,'percent10M_secondD_'+name+'_classnum.pickle')
        percent10_path_pickle_dir = os.path.join(pickle_dir2,'percent10M_secondD_'+name+'_path.pickle')
        with open(percent10_confidence_pickle_dir , 'wb') as f:
            pickle.dump(tensor2, f)
        with open(percent10_classnum_pickle_dir, 'wb') as f:
            pickle.dump(pre_result2, f)
        with open(percent10_path_pickle_dir, 'wb') as f:
            pickle.dump(path2, f)

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("product second5 cofidence completed time: ", "{:.0f}".format(elapsed_time))

#製作min model要用的label和data( 只取最大的x個值 )
#製作min model要用的label和資料集( 只取最大的x個值 )
#label類別 0:上升 1:不變 2:下降
with open(os.path.join(pickle_dir,'first5M_secondD_train_confidence.pickle'), 'rb') as f:
    train_data = pickle.load(f)
with open(os.path.join(pickle_dir,'percent10M_secondD_train_confidence.pickle'), 'rb') as f:
    train_label = pickle.load(f)
with open(os.path.join(pickle_dir,'first5M_secondD_val_confidence.pickle'), 'rb') as f:
    val_data = pickle.load(f)
with open(os.path.join(pickle_dir,'percent10M_secondD_val_confidence.pickle'), 'rb') as f:
    val_label = pickle.load(f)
for class_ in range(class_number):
    t_label,t_data,v_label,v_data  = [ []for x in range(4)]
    save_pickle_dir = os.path.join(pickle_dir,'min_model',str(class_))
    os.makedirs(save_pickle_dir, exist_ok=True)
    for t in range(len(train_data)):
        value , index = torch.sort(train_data[t].squeeze(),descending = True)
        for t2 in range(int(len(index)/5)):
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
        for t2 in range(int(len(index)/5)):
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

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("produce min model data completed time: ", "{:.0f}".format(elapsed_time))

#定義min model分類模型
device="cuda" if torch.cuda.is_available() else "cpu"
MyResModel = main_model.LSTM_FCN(input_dim=1, hidden_dim=32, output_dim=3, layers=1).to(device)
MyResModel.init_model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(MyResModel.parameters(), lr=0.001, momentum=0.8)

#train min model分類模型
#參數設定
Epochs=100
max_test_acc=0.0
batch_size = 32
good_model_list = []
device = 'cuda'
save_min_model_dir = '../all_file/min_model_weight'
os.makedirs(save_min_model_dir,exist_ok=True)

#匯入所有label
train_label_ = []
for t in range(class_number ):
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_train_label.pickle'), 'rb') as f:
        train_label_.append(pickle.load(f))
val_label_ = []
for t in range(class_number ):
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_val_label.pickle'), 'rb') as f:
        val_label_.append(pickle.load(f))

#匯入所有confidence
train_data_ = []
for t in range(class_number ):
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_train_data.pickle'), 'rb') as f:
        train_data_.append(pickle.load(f))
val_data_ = []
for t in range(class_number ):
    with open(os.path.join(save_pickle_dir,'class'+str(class_)+'_val_data.pickle'), 'rb') as f:
        val_data_.append(pickle.load(f))

#訓練各類模型
for class_n in range(class_number ):
    print('class '+str(class_n)+' model')
    # 按batch size包裝training set資料
    register_1,register_2,train_tensor_list,train_label_list = [[] for x in range(4)]
    count = 0
    best_acc = 0
    for t in range(len(train_data_[class_n])):
        count+=1

        #不做confidence calibration
        numpy = FUN.softmax(Variable(train_data_[class_n][t])).data.cpu().numpy()#把input從confidence轉為機率

        #做confidence calibration
        # caliration = train_data_[class_n][t]/0.5
        # soft = FUN.softmax(Variable(caliration)).data.cpu()#把input從confidence轉為機率
        # numpy = soft.numpy()

        # 對input做confidence calibration( 只取confidence高的 )
        # value , index = torch.sort(train_data_[class_n][t].squeeze(),descending = True)
        # for t2 in range(len(index)):
        #     if t2>20:
        #         pass
        #     else:
        #         numpy[0][index[t2]] = numpy[0][index[t2]]


        tensor = torch.tensor(numpy)
        register_1.append(tensor)
        register_2.append(train_label_[class_n][t])
        if count % batch_size ==0:
            a = torch.stack(register_1)
            b = torch.stack(register_2)
            train_tensor_list.append(a)
            train_label_list.append(b)
            register_1 = []
            register_2 = []
    #依bach size打包validation set
    register_1,register_2,val_tensor_list,val_label_list = [[] for x in range(4)]
    count = 0
    for t in range(len(val_data_[class_n])):
        count+=1

        #不做confidence calibration
        numpy = FUN.softmax(Variable(val_data_[class_n][t])).data.cpu().numpy()#把input從confidence轉為機率

        #做confidence calibration
        # caliration = val_data_[class_n][t]/0.5
        # soft = FUN.softmax(Variable(caliration)).data.cpu()#把input從confidence轉為機率
        # numpy = soft.numpy()

        # 對input做confidence calibration( 只取confidence高的 )
        # value , index = torch.sort(val_data_[class_n][t].squeeze(),descending = True)
        # for t2 in range(len(index)):
        #     if t2>20:
        #         pass
        #     else:
        #         numpy[0][index[t2]] = numpy[0][index[t2]]

        tensor = torch.tensor(numpy)
        register_1.append(tensor)
        register_2.append(val_label_[class_n][t])
        if count % batch_size ==0:
            a = torch.stack(register_1)
            b = torch.stack(register_2)
            val_tensor_list.append(a)
            val_label_list.append(b)
            register_1 = []
            register_2 = []

    acc_register = 0
    #開始訓練模型
    for epoch in range(Epochs):
        optimizer1 = function.lrfn(epoch,optimizer)
        train_loss=0.0
        train_acc=0.0
        test_loss=0.0
        test_acc=0.0
        MyResModel.train()
        count = 0
        train_loss_count = 0
        train_acc_count = 0
        for data_ in train_tensor_list:
            optimizer1.zero_grad()
            pred = MyResModel(data_.to(device))
            max_,class_ = torch.max(pred.data,1)
            train_correct = (class_==train_label_list[count].to(device)).sum()
            train_acc = train_correct / batch_size
            train_acc_count = train_acc_count + train_acc.item()
            loss = criterion(pred.to(device),train_label_list[count].to(device))
            train_loss_count = train_loss_count+loss.item()
            count+=1
            loss.backward()
            optimizer1.step()
        MyResModel.eval()
        a = (epoch+1) / 100
        if count ==0 :
            count = 1
        if (epoch+1) % 10 ==0:
            pass
            print('epoch : '+ str(int(a*100)) +' train_loss : '+str(train_loss_count/count)+' train_Acc : '+str(train_acc_count/count))
        #算驗證集loss和acc
        if (epoch+1) % 100 ==0:
            count = 0
            loss_count = 0
            acc_count = 0
            for data_ in val_tensor_list:
                pred = MyResModel(data_.to(device))
                max_,class_ = torch.max(pred.data,1)
                correct = (class_==val_label_list[count].to(device)).sum()
                acc = correct / batch_size
                acc_count = acc_count + acc.item()
                loss = criterion(pred.to(device),val_label_list[count].to(device))
                loss_count = loss_count+loss.item()
                count+=1
            if count ==0 :
                count = 1
            if acc_count/count >= best_acc:
                best_acc = acc_count/count
                torch.save(MyResModel.state_dict(),os.path.join (save_min_model_dir,'model'+str(class_n)+'.pth'))
                a = (epoch+1) / 100
                print('epoch : '+ str(a*100) +' loss : '+str(loss_count/count)+' Acc : '+str(acc_count/count))
                acc_register = acc_count/count

    #紀錄min model中準確率好的class      
    if acc_register >=0.3:
        good_model_list.append(str(class_n))
os.makedirs(os.path.join(pickle_dir,'good_model'),exist_ok=True)
with open(os.path.join(pickle_dir,'good_model','good_model_list.pickle'), 'wb') as f:
    pickle.dump(good_model_list, f)
with open(os.path.join(pickle_dir,'good_model','good_model_list.pickle'), 'rb') as f:
    good_model = pickle.load(f)
print('min model have '+str(len(good_model))+' acc over target')

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("min model training completed time: ", "{:.0f}".format(elapsed_time))

#用min model修改10% model產出的confidence(修改前五大)
with open(percent10_confidence_pickle_dir, 'rb') as f:  
    result = pickle.load(f)
with open( percent10_classnum_pickle_dir, 'rb') as f:
    label = pickle.load(f)
min_model_list = os.listdir(save_min_model_dir)
min_model_file = []
for t in range(len(min_model_list)):
    min_model_path = os.path.join(save_min_model_dir,min_model_list[t])
    min_model_file.append(min_model_path)
for t in range(len(result)):     
    value , index = torch.sort(result[t].squeeze(),descending = True)
    for t2 in range(int(len(index)/20)):
        MyResModel.load_state_dict(torch.load(min_model_file[int(index[t2])]))
        MyResModel = MyResModel.to(device)
        numpy = FUN.softmax(Variable(result[t])).data.cpu().numpy()
        tensor = torch.tensor(numpy)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        output = MyResModel(tensor)
        pre_class = torch.argmax(output)
        #只修改模型效果好的
        if index[t2] in good_model:
            if int(pre_class)==0:
                result[t][0][int(index[t2])] = result[t][0][int(index[t2])] + result[t][0][int(index[t2])]*0.5
            if int(pre_class)==1:
                result[t][0][int(index[t2])] = result[t][0][int(index[t2])]
            if int(pre_class)==2:
                result[t][0][int(index[t2])] = result[t][0][int(index[t2])] - result[t][0][int(index[t2])]*0.2
acc_count = 0
for t in range(len(result)):        
        r = torch.argmax(result[t])
        if int(r) == label[t]:
            acc_count+=1
print('after min model revise acc : '+str(acc_count/len(result)))
save_revise_confidenxe_dir = os.path.join(pickle_dir,'revise_confidence')
os.makedirs(save_revise_confidenxe_dir ,exist_ok=True)
save_revise_confidenxe_path = os.path.join(save_revise_confidenxe_dir,'revise_confidence.pickle')
with open(save_revise_confidenxe_path, 'wb') as f:
    pickle.dump(result, f)   

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("min model revise confidence completed time: ", "{:.0f}".format(elapsed_time))

#用改完的confidence算enstropy挑 x% data(各類數量平均)
with open( percent10_path_pickle_dir , 'rb') as f:
    path = pickle.load(f)
with open( save_revise_confidenxe_path , 'rb') as f:
    result = pickle.load(f)
img_save_dir = '../all_file/selection_data' 
all_en = []
for t in range(len(result)):
    odds = FUN.softmax(Variable(result[t]).cpu()).data.numpy()
    odds = odds.reshape([class_number])
    data_entropy = function.entropy(odds)
    all_en.append(data_entropy)
sort = sorted(range(len(all_en)) , reverse = True,key = lambda k : all_en[k])
img_dir_list=glob.glob(percent90_dir+os.sep+"*")#獲取每個類别所在的路徑（一個類别對應一個文件夾）
for class_dir in img_dir_list:
    class_name=class_dir.split(os.sep)[-1] #獲取當前類别
    save_train=img_save_dir+os.sep+os.sep+class_name
    os.makedirs(save_train,exist_ok=True)#建立對應的文件夾
con = [0 for t in range(class_number)] #創一個維度等於模型類別數的list
for t in range(int(len(sort))):
    re_path = str(path [sort[t]]).replace('\\','/')
    re_path = re_path.replace('('')''\,','')
    s_path = re_path.rsplit('/',2) #把照片所在的資料夾名和檔名切出來
    print('--------debug--------')
    print(re_path)
    print(s_path[1])
    print(s_path[2])
    print('--------debug end--------')
    if con[int(s_path[1])] <= (int(len(sort))/class_number) *0.5: #控制每一類數量平均
        con[int(s_path[1])] = con[int(s_path[1])]+1
        img = cv2.imread (path[sort[t]])
        cv2.imwrite(img_save_dir+'/'+ s_path[1]+'/'+s_path[2], img)    

#///計算執行時間///
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print("select target quantity data completed time: ", "{:.0f}".format(elapsed_time))