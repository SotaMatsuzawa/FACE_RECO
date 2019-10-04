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
  0:"gyaku",
  1: "home",
  2: "maru",
  3:"shikaku",
  4:"tamago"
}

#指定した画像(img_path)を学習結果(ckpt_path)を用いて判定する
def evaluation(img_path, ckpt_path):
    # GraphのReset
    tf.compat.v1.reset_default_graph()#new verに変更
    # 画像を開く
    f = open(img_path, 'r')
    # 画像読み込み
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # モノクロ画像に変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, 1.1, 3)
    if len(face) > 0:
        for rect in face:
            # 加工した画像に何でもいいので適当な名前をつけたかった。日付秒数とかでいいかも
            date_str = str(datetime.datetime.now())
            date_str=date_str.replace('.','-').replace(':',"-")
            # 顔部分を赤線で書こう
            cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)
            # 顔部分を赤線で囲った画像の保存先
            face_detect_img_path = r'C:\Users\souta\Desktop\FACE_RECO\static\images\face_detect/' + date_str + '.jpg'
            # 顔部分を赤線で囲った画像の保存
            cv2.imwrite(face_detect_img_path, img)
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            # 検出した顔を切り抜いた画像を保存
            cv2.imwrite(r'C:\Users\souta\Desktop\FACE_RECO\static\images\cut_face/' + date_str + '.jpg', img[y:y+h, x:x+w])
            # TensorFlowへ渡す切り抜いた顔画像
            target_image_path = r'C:\Users\souta\Desktop\FACE_RECO\static\images\cut_face/' + date_str + '.jpg'
    else:
        # 顔が見つからなければ処理終了
        print("image:NoFace")
        return
    f.close()

    f = open(target_image_path, 'r')
    # データを入れる配列
    image = []
    # 画像読み込み
    img = cv2.imread(target_image_path)
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
    #ans=rank[0]
    #print(ans["label"],ans["name"])
    return [rank, face_detect_img_path, target_image_path]    
    """  
    ---------------------------------------------
    """
# コマンドラインからのテスト用
if __name__ == '__main__':
  evaluation(r'C:\Users\souta\Desktop\FACE_RECO\static\images\default\sample.jpg', r'C:\Users\souta\Desktop\FACE_RECO\model_ver1\model1.ckpt')