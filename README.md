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
`utils.py` : Read images from the specified directory , resize them and pickle them.

### Train model
`train.py` : The main part of training model. Make model instance and dataloaders, and then train model.

- `dataset.py` : Dataset class for training and evaluation and classes for Data Augmentation.

- `model.py` : Image classifier. ResNet、DenseNet、DualPathNetworks


### Evaluate model
`test.py` : Evaluate model performance using pickled tst dataset.

### Infer images
`infer.py` : Evaluate model (path to model and path to images are needed)


### Utils 
`utils.py`



## Usage
If you have images at `images/` and trained model at `model/` and wanna classify the images using threshold `0.467`, you can infer as following.
```
python infer.py --model_path model/valloss0.06944_epoch68_img300.model --images_path images/ --image_size 300 --threshold 0.467
```

