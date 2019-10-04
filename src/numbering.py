# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 23:56:28 2019

@author: souta
"""

import cv2


list=["\\gyaku","\\home","\\maru","\\shikaku","\\tamago"]

img_path=r"C:\Users\souta\Desktop\data\train"


save_path=r"C:\Users\souta\Desktop\FACE_RECO\data\train"

image_count=500

for name in list:
    img_num=0
    save_num=0
    for i in range(image_count):
        img = cv2.imread(img_path+name+"\\"+str(img_num)+".jpg", cv2.IMREAD_COLOR)
        img_num+=1
        if img is None:
             print('image' +str(img_num)+".jpg" + ':No')
        else:
            save_num+=1
            cv2.imwrite(save_path+name+"/"+name+"_"+str(save_num)+".jpg",img)

print("finish")    