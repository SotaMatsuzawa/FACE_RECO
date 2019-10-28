# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:45:30 2019

@author: souta
"""



"""
このファイルはモデルを再利用して
画像を入力として
判定結果と切り取った顔画像を出力する



"""

import numpy as np
import cv2
import tensorflow as tf
import random
import datetime
import inference as inf

# OpenCVのデフォルトの顔の分類器のpath
cascade_path = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

# 識別ラベルと各ラベル番号に対応する名前
FACE_TYPES = {
  0:"tamago",
  1: "gyaku",
  2: "maru",
  3:"home",
  4:"shikaku"
}

#指定した画像(img_path)を学習結果(ckpt_path)を用いて判定する
def evaluation(img_path, ckpt_path,i):
    # GraphのReset
    tf.compat.v1.reset_default_graph()#new verに変更
    # 画像を開く
    f = open(img_path, 'r')
    # TensorFlowへ渡す切り抜いた顔画像
    target_image_path = img_path
    f.close()

    f = open(target_image_path, 'r')
    # データを入れる配列
    image = []
    # 画像読み込み
    img = cv2.imread(target_image_path)
    img_pre = cv2.imread(target_image_path)    
    # 28px*28pxにリサイズ
    img = cv2.resize(img, (28, 28))
    # 画像情報を一列にした後、0-1のfloat値にする
    image.append(img.flatten().astype(np.float32)/255.0)
    # numpy形式に変換し、TensorFlowで処理できるようにする
    image = np.asarray(image)
    # 入力画像に対して、各ラベルの確率を出力して返す(main.pyより呼び出し)
    logits = inf.inference(image, 1.0)
    # We can just use 'c.eval()' without passing 'sess'
    sess = tf.compat.v1.InteractiveSession()
    # restore(パラメーター読み込み)の準備
    saver = tf.compat.v1.train.Saver()
    # 変数の初期化
    sess.run(tf.initialize_all_variables())
    if ckpt_path:
        # 学習後のパラメーターの読み込み
        saver.restore(sess, ckpt_path)
    # sess.run(logits)と同じ
    softmax = logits.eval()
    # 判定結果
    result = softmax[0]
    # 判定結果を%にして四捨五入
    rates = [round(n * 100.0, 1) for n in result]
    humans = []
    # ラベル番号、名前、パーセンテージのHashを作成
    for index, rate in enumerate(rates):
        name = FACE_TYPES[index]
        humans.append({
            'label': index,
            'name': name,
            'rate': rate})
    # パーセンテージの高い順にソート
    rank = sorted(humans, key=lambda x: x['rate'], reverse=True)
    
    # 判定結果と加工した画像のpathを返す
    """
    ---------------------------------------------
    """
    print(rank)
    ans=rank[0]
    #print(ans["label"],ans["name"])
    cv2.imwrite(r'C:\Users\souta\Desktop\FACE_RECO\static\images'+"\\"+ans["name"]+"\\"+'gyaku'+str(i)+'.jpg', img_pre)
    
    return [rank,target_image_path]    
    """  
    ---------------------------------------------
    """
# コマンドラインからのテスト用
if __name__ == '__main__':
    for i in range(332):
        path=r'C:\Users\souta\Desktop\FACE_RECO\data_4\test\gyaku\gyaku'+str(i)+'.jpg'
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        if img is None:
            print(path+":None")
        else:
            evaluation(path, r'C:\Users\souta\Desktop\FACE_RECO\model_ver4\model4.ckpt',i)