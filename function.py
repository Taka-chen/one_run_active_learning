#定義function


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


#對影像做旋轉和亮度,對比度,飽和度的調整
def expend1_img(img):
    a = random.random()
    t2 = transforms.RandomHorizontalFlip(p=0.5)  # 水平镜像，p是機率
    t3 = transforms.RandomVerticalFlip(p=0.2) #垂直鏡像
    img = t2(img)
    img = t3(img)
    if a<0.4:
        t1 = transforms.RandomRotation(45)  # 随機旋轉，旋轉範圍為【-45,45】
        img = t1(img)
    t4 = transforms.ColorJitter(brightness=(0.8,1.5), contrast=(0.8,1.5), saturation=(0.8,1.5))#調整亮度,對比度,飽和度
    img = t4(img)
    return img


#對圖片做透視轉換
def expend2_img(img):
    t = transforms.RandomPerspective(distortion_scale=0.6,p=1,interpolation = 2,fill=0) #圖片透視化
    img2 = t(img)
    return img2


#切分訓練集和測試集，並進行補邊處理
def split_train_test(img_dir,save_dir,train_val_num):
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
        save_train=save_dir+os.sep+'train'+os.sep+class_name
        save_val=save_dir+os.sep+"val"+os.sep+class_name
        os.makedirs(save_train,exist_ok=True)
        os.makedirs(save_val,exist_ok=True) #建立對應的文件夾
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

