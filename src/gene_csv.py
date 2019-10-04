# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:02:57 2019

@author: souta
"""


"""
このファイルはtensorflowに読み込ませるデータのラベル付けを
txtファイルとして保存するためのもの
ラベルは
逆三角顔：０
ホームベース顔：１
丸顔：２
四角顔：３
卵顔：４


"""

import cv2
import csv
path_list=["\\gyaku","\\home","\\maru","\\shikaku","\\tamago"]
path_dir={"\\gyaku":0,"\\home":1,"\\maru":2,"\\shikaku":3,"\\tamago":4}

t=["\\train","\\test"]
#img_path=r"C:\Users\souta\Desktop\Face"

write_path=r"C:\Users\souta\Desktop\FACE_RECO\data"
path_train=r"C:\Users\souta\Desktop\FACE_RECO\data\train\data.csv"
path_test=r"C:\Users\souta\Desktop\FACE_RECO\data\test\data.csv"


for t_type in t:#train or test
    with open(write_path+t_type+"\\data.txt",'w',newline="") as f:
        writer=csv.writer(f)
        for i in range(0,231):#img_num
            for j in range(5):#face_type
                path=write_path+t_type+path_list[j]+path_list[j]+str(i)+".jpg"
                img = cv2.imread(path,cv2.IMREAD_COLOR)
                if img is None:
                    print(path+":None")
                else:
                    writer.writerow([path,j])
                    
        
print("finish")

