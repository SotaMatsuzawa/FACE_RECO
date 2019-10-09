# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 22:42:29 2019

@author: souta
"""


"""
このファイルはデータセットをトレーニングデータとテストデータに分けるためのもの

だいたい
トレーニング：テスト＝3：1
（4の倍数でテストデータにしている）

"""


import cv2
path_list=["\\gyaku_FACE","\\homebase_FACE","\\maru_FACE","\\sikaku_FACE","\\tamago_FACE"]
save_list=["\\gyaku","\\home","\\maru","\\shikaku","\\tamago"]
img_path=r"C:\Users\souta\Desktop\Face_RECO"


save_path_train=r"C:\Users\souta\Desktop\FACE_RECO\data\train"
save_path_test=r"C:\Users\souta\Desktop\FACE_RECO\data\test"

image_count=500

for l in range(len(path_list)):
    img_num=0
    save_num=0
    for i in range(image_count):
        img = cv2.imread(img_path+path_list[l]+"\\cutted"+str(img_num)+".jpg", cv2.IMREAD_COLOR)
        img_num+=1
        if img is None:
             print('image' + img_path+str(img_num)+".jpg" + ':No')
        elif img_num%4==0:
            save_num+=1
            cv2.imwrite(save_path_test+save_list[l]+"/"+save_list[l]+str(save_num)+".jpg",img)
        else:
            save_num+=1
            cv2.imwrite(save_path_train+save_list[l]+"/"+save_list[l]+str(save_num)+".jpg",img)
    
print("finish")    