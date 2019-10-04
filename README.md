# FACE_RECO

## Name
Face Recognition

## Overview
顔画像からおすすめの髪形をレコメンドするサービスを大学の授業で作ることになり、
機械学習モデルの作成部分を担当した。

主に、ディープラーニングでザッカーバーグの顔を識別するAIを作る（https://qiita.com/AkiyoshiOkano/items/72f3e4ba9caf514460ee）
を参考にさせていただいた。

行った手順は、  
１．顔画像をスクレイピングしてもらって、データセットを用意してもらう<br>
２．画像データから顔のみを切り取る（openCV）<br>
３．トレーニングデータとテストデータに分ける<br>
４．トレーニングデータでモデルを学習させ、テストデータで性能を評価する<br>
である。  


### ディレクトリの構造
/FACE_RECO
  /data
    /train
     /test  
  /src  
    main.py モデルを回すためのmain部分、モデルを作成し保存する  
    eval.py　保存したモデルを使って新しい画像の判定をする  
    face_reco.py　画像から顔を切り取って顔のみの画像を保存する  
    gene_csv.py　データのラベリングを行うためのtxtファイルを出力する  
    inference.py　モデルの学習部分  
    train_test_split.py　顔のみの画像をトレーニングデータとテストデータに分ける  
  /static  
    /images  
      /cut_face  
      /default  
      /face_detect  
      
    







