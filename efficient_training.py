#efficientionnet訓練模型
import function

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
        base_model = EfficientNet.from_name('efficientnet-b5') #加載模型，使用b幾的就改為b幾
        state_dict = torch.load(self.weights)
        base_model.load_state_dict(state_dict)
        # 修改全連接層
        num_ftrs = base_model._fc.in_features
        base_model._fc = nn.Linear(num_ftrs, self.class_num)
        print(device)
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
            # print(run_correct,'------------',count)
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
                start_time=(time.strftime("%m%d",time.localtime()))
                save_weight=self.save_dir+os.sep+start_time #保存路徑
                os.makedirs(save_weight,exist_ok=True)
                torch.save(self.model.state_dict(), save_weight + os.sep + self.save_model_name)#不加state_dict()存法會直接把模型架構和權重一起存入weight檔中
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