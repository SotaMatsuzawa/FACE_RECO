
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 18:47:45 2019

@author: souta
"""

"""
このファイルは画像を入力とし、顔のみを切り取った画像をjpgファイルとして保存する
"""

import cv2

#画像の読み込みから表示まで
face_cascade = cv2.CascadeClassifier(r'C:\Users\souta\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

img_path_0=r"C:\Users\souta\Desktop\FACE_RECO\gyakusankaku_yama_2/000"
img_path_1=r"C:\Users\souta\Desktop\FACE_RECO\gyakusankaku_yama_2\0000"
img_path_2=r"C:\Users\souta\Desktop\FACE_RECO\gyakusankaku_yama_2\00000"



save_path=r"C:\Users\souta\Desktop\FACE_RECO\FACE_3\gyaku_FACE/"
#集めた画像の枚数
image_count=129
#顔検知に成功した数(def=0)
face_detect_count=105
img_num=1

for i in range(image_count):
    if img_num<10:
        img_path=img_path_2
    elif img_num<100 and img_num>=10:
        img_path=img_path_1
    else:
        img_path=img_path_0
    img = cv2.imread(img_path+str(img_num)+".jpg", cv2.IMREAD_COLOR)
    img_num+=1
    if img is None:
         print('image' + img_path+str(img_num)+".jpg" + ':NoFace')
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 3)
        if len(face) > 0:
            for rect in face:
                # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
                # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
                # cv2.imwrite('detected.jpg', img)
                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]
                cv2.imwrite(save_path + str(face_detect_count) + '.jpg',img[y:y+h,  x:x+w])
                face_detect_count = face_detect_count + 1
        else:
            print('image' +img_path+ str(img_num) + ':NoFace') 
    
print("顔画像の切り取り作業終了")     