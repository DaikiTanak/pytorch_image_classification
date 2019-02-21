# pytorch_image_classification


## Requirements
```
numpy                              1.15.1    
opencv-python                      3.4.4.19      
pandas                             0.23.4    
Pillow                             5.2.0     
scikit-image                       0.14.0    
scikit-learn                       0.19.2    
scipy                              1.1.0     
torch                              1.0.0     
torchvision                        0.2.1          
tqdm                               4.26.0 
Python                             3.7.0
```

## Files
### Preprocess images
`utils.py` : 画像が格納されているフォルダから画像を読み、リサイズしてデータのpickleを作成する。

### Train model
`train.py` : モデル訓練の主要部。モデルインスタンス作成、データローダー作成、モデル訓練を行う。

- `dataset.py` : 訓練や推論に用いるDataset、Data Augmentationに使う各種クラス。

- `model.py` : 分類器。ResNet、DenseNet、DualPathNetworks


### Evaluate model
`test.py` : pickleしたテストデータを使って性能を評価する。

### Infer images
`infer.py` : 画像へのパス、モデルへのパスを指定して推論を行う関数


### Utils 
`utils.py`



## Usage
If you have images at `images/` and trained model at `model/` and wanna classify the images using threshold `0.467`, you can infer as following.
```
python infer.py --model_path model/valloss0.06944_epoch68_img300.model --images_path images/ --image_size 300 --threshold 0.467
```

