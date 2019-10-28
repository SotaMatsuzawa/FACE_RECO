# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 18:18:35 2019

@author: souta
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:02:57 2019

@author: souta
"""


"""

data_5に保存するるver

このファイルはtensorflowに読み込ませるデータのラベル付けを
txtファイルとして保存するためのもの
ラベルは
卵顔：０
逆三角：１
丸顔：２
ホームベース顔：３
四角顔：４


"""

import cv2
import csv
path_list=["\\tamago","\\gyaku","\\home"]
path_dir={"\\tamago":0,"\\gyaku":1,"\\home":2}

t=["\\train","\\test"]
#img_path=r"C:\Users\souta\Desktop\Face"

write_path=r"C:\Users\souta\Desktop\FACE_RECO\data_6"
path_train=r"C:\Users\souta\Desktop\FACE_RECO\data_6\train\data.csv"
path_test=r"C:\Users\souta\Desktop\FACE_RECO\data_6\test\data.csv"


for t_type in t:#train or test
    with open(write_path+t_type+"\\data.txt",'w',newline="") as f:
        writer=csv.writer(f)
        for i in range(0,413):#img_num
            for j in range(3):#face_type
                path=write_path+t_type+path_list[j]+path_list[j]+str(i)+".jpg"
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                if img is None:
                    print(path+":None")
                else:
                    writer.writerow([path,j])
                    
        
print("finish")

