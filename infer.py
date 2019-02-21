import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter, ImageDraw
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict
import copy
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms, utils
from PIL import Image, ImageFilter, ImageDraw, ImageChops

from collections import Counter
from datetime import datetime, timezone, timedelta
import random

from model import SimpleNet, se_resnet18, se_resnet152, densenet121, densenet169, densenet201, dpn92, dpn131, dpn98
from dataset import *
from utils import get_roc_auc_curve, get_precision_recall_curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import argparse
from scipy.ndimage.interpolation import rotate
import os
import time
from joblib import Parallel, delayed


parser = argparse.ArgumentParser(
            prog='infer.py',
            usage='infer images',
            description='description',
            epilog='end',
            add_help=True,
            )
parser.add_argument('-g', '--gpu',
                    help="gpu id to run the model.",
                    default=0,
                    type=int,
                    required=False)
parser.add_argument('-b', '--batchsize',
                    help="num of data in each mini-batch",
                    default=16,
                    type=int,
                    required=False)
parser.add_argument('--model_path',
                    help="path to model to use for inference",
                    default="model/densenet121.model",
                    type=str,
                    required=True)
parser.add_argument('--images_path',
                    help="path to images to infer",
                    default="images/",
                    type=str,
                    required=True)
parser.add_argument('--image_size',
                    help="size of image : this must be the same size as that of training.",
                    default=300,
                    type=int,
                    required=False)
parser.add_argument('--threshold',
                    help="threshold to determine prediciton class",
                    default=0.5,
                    type=float,
                    required=False)

args = parser.parse_args()
assert os.path.exists(args.model_path), "There is not such a file : {}".format(args.model_path)
assert os.path.exists(args.images_path), "There is not such a directory : {}".format(args.images_path)

gpu_name = "cuda:" + str(args.gpu)
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
BATCH_SIZE = args.batchsize

print("="*15 + " inference config " + "="*15)
print("device: ", device)
print("batch size: ", BATCH_SIZE)
print("image size: ", args.image_size)
print("model path: ", args.model_path)
print("images path: ", args.images_path)
print("="*40)

print("open images ...")

image_names = os.listdir(args.images_path)

def open_image(image_name, images_path, image_size):
    assert image_name[-4:] == ".jpg" or image_name[-4:] == ".png" or image_name[-5:] == ".jpeg"

    image_path = os.path.join(images_path, image_name)
    image = Image.open(image_path)

    if not len(np.array(image)) == 3:
        image = image.convert("RGB")

    img_resized = image.resize((image_size, image_size), Image.LANCZOS)
    img_resized_array = np.array(img_resized, dtype="uint8")
    return (img_resized_array, image_name)

image_information = Parallel(n_jobs=-1, verbose=-1)([delayed(open_image)(name, args.images_path, args.image_size) for name in image_names])
image_holder, imagename_holder = [], []

for (image_array, image_name) in image_information:
    image_holder.append(image_array)
    imagename_holder.append(image_name)


model = densenet121(if_mixup=False, if_selayer=True, first_conv_stride=2, first_pool=True).to(device)

checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("==>" + "model loaded:", args.model_path)

ch_mean = [237.78516378, 237.16343756, 237.0501237]
ch_std = [146.47616225347468, 146.6169214951974, 145.59586636818233]

# dummy labels
labels = [0]*len(image_holder)

test = Dataset(image_holder, labels, paths=imagename_holder, transform=transforms.Compose([
                                                                                          Regularizer(ch_mean, ch_std=ch_std),
                                                                                          ToTensor()]))
test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

print("start inference ...")
time_list = []
with torch.no_grad():
    model.eval()
    y_score = []
    y_pred = []
    img_path = []
    y = []

    for idx, data in enumerate(tqdm(test_iter)):
        now = time.time()
        path, inputs, labels = data
        img_path.extend(path)
        labels.to(device)

        inputs = inputs.to(device)
        outputs = model(inputs)
        max_index = outputs.max(dim = 1)[1]
        y_score.extend(outputs.cpu().numpy()[:, 1])
        y_pred.extend(max_index.cpu().numpy())

        time_list.append(time.time()-now)

# print("mean inference time per image: ", np.mean(time_list))
# print("std inference time per image: ", np.std(time_list))


def sigmoid(x):
  return 1 / (1+np.exp(-x))

for score, name in zip(y_score, img_path):
    if sigmoid(score) < args.threshold:
        pred_class = "No Date"
    else:
        pred_class = "Date"
    print("image_names:{}, class:{}".format(name, pred_class))

# end
