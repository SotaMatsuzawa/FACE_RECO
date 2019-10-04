# FACE_RECO

## Name　名前
Face Recognition

## Overview 概要
顔画像からおすすめの髪形をレコメンドするサービスを大学の授業で作ることになり、
機械学習モデルの作成部分を担当した。

主に、
 [ディープラーニングでザッカーバーグの顔を識別するAIを作る](https://qiita.com/AkiyoshiOkano/items/72f3e4ba9caf514460ee) 
を参考にさせていただいた。

行った手順は、  
１．顔画像をスクレイピングしてもらって、データセットを用意してもらう<br>
２．画像データから顔のみを切り取る（openCV）<br>
３．トレーニングデータとテストデータに分ける<br>
４．トレーニングデータでモデルを学習させ、テストデータで性能を評価する<br>
である。  


### Directory structure ディレクトリの構造
/FACE_RECO<br>
　　/data<br>
　　　　/train<br>
　　　　/test<br>
　　/src<br>
    　　-main.py モデルを回すためのmain部分、モデルを作成し保存す<br>
    　　-eval.py　保存したモデルを使って新しい画像の判定をする<br>
    　　-face_reco.py　画像から顔を切り取って顔のみの画像を保存する<br>
    　　-gene_csv.py　データのラベリングを行うためのtxtファイルを出力する<br>
    　　-inference.py　モデルの学習部分<br>
    　　-train_test_split.py　顔のみの画像をトレーニングデータとテストデータに分ける<br>
　　/static<br>
    　　/images<br>
      　　　　/cut_face<br>
      　　　　/default<br>
      　　　　/face_detect<br>
      
    







