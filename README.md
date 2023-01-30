# one_run_active_learning

## 檔案功能
* ### main.py  主程式
* ### main_model.py  主要的影像分類模型
* ### function.py  會用到的自定義函式
* ### min_model.py cofidence預測模型

## 執行步驟
* ### 在code上一層資料夾中創一個檔名為'all_file'的資料夾，並把資料集放入其中
* * ### 資料集格式 : 把照片放入對應的類別的資料夾中，並把所有類別的資料夾放入一個主資料夾'data'中
* ### 修改main.py的img_dir為資料集的路徑
* ### 修改main.py的class_num為欲預測的類別數
* ### 準備預訓練的efficientnet b5檔
* ### 預訓練權重放到路徑'all_file/model下'(範例模型使用efficient net b5)
* * ### 範例模型使用的預訓練權重檔名為 efficientnet-b5-b6417697.pth 若使用的權重檔名不同則須將main.py中出現上述檔名的部分進行修改


