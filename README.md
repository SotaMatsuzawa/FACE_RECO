# FACE_RECO

##Name
Face Recognition

##Overview
顔画像からおすすめの髪形をレコメンドするサービスを大学の授業で作ることになり、
機械学習モデルの作成部分を担当した。

主に、ディープラーニングでザッカーバーグの顔を識別するAIを作る（https://qiita.com/AkiyoshiOkano/items/72f3e4ba9caf514460ee）
を参考にさせていただいた。

行った手順は、  
１．顔画像をスクレイピングしてもらって、データセットを用意してもらう  
２．画像データから顔のみを切り取る（openCV）  
３．トレーニングデータとテストデータに分ける　　
４．トレーニングデータでモデルを学習させ、テストデータで性能を評価する　
である。

###ディレクトリの構造
/FACE_RECO
  /data
    /train
    /test
  /src
    main.py
    eval.py
    face_reco.py
    gene_csv.py
    inference.py
    train_test_split.py
  /static
    /images
      /cut_face
      /default
      /face_detect
      
    







