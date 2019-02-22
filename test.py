import numpy as np
import pandas as pd
import os
from PIL import Image
import copy
from tqdm import tqdm
import shutil

import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms, utils

from collections import Counter

from model import SimpleNet, se_resnet18, se_resnet152, densenet121, densenet169, densenet201, dpn92, dpn131, dpn98, __all__
from dataset import *
from utils import get_roc_auc_curve, get_precision_recall_curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

import argparse
from scipy.ndimage.interpolation import rotate

def TTA_for_img(img_tensor):
    # Test time augmentation function for an image.
    # Rotate, color change
    def rotate_image(img):
        # img is numpy,array

        H,W = img.shape[:2]

        rotated_list = [torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0)]

        rotate90 = torch.tensor(rotate(img, 90), dtype=torch.float).permute(2,0,1).unsqueeze(0)
        rotate180 = torch.tensor(rotate(img, 180), dtype=torch.float).permute(2,0,1).unsqueeze(0)
        rotate270 = torch.tensor(rotate(img, 270), dtype=torch.float).permute(2,0,1).unsqueeze(0)

        rotated_list.append(rotate90)
        rotated_list.append(rotate180)
        rotated_list.append(rotate270)

        return rotated_list

    img_tensor = img_tensor.permute(1,2,0)
    tensor_list = rotate_image(img_tensor.numpy())

    # B*C*H*W
    tensor = torch.cat(tensor_list, dim=0)

    return tensor

def tensor_to_pil(tensor):

    original = tensor.cpu() * 255.0 + torch.tensor([237.76504679, 237.21081107, 237.08555321]).unsqueeze(0).expand(300,300,3).permute(2,0,1)
    trans = transforms.ToPILImage()
    pil_img = trans(original)

    return pil_img



parser = argparse.ArgumentParser(
            prog='test.py', # プログラム名
            usage='Evaluating Classifier', # プログラムの利用方法
            description='description', # 引数のヘルプの前に表示
            epilog='end',
            add_help=True, # -h/–help オプションの追加
            )
parser.add_argument('-g', '--gpu',
                    help="gpu id to run the model at.",
                    default=0,
                    type=int,
                    required=False)
parser.add_argument('-b', '--batchsize',
                    help="num of data in each mini-batch",
                    default=16,
                    type=int,
                    required=False)

parser.add_argument('--first_stride',
                    help="stride width in first conv",
                    default=2,
                    type=int,
                    required=False)

parser.add_argument('--directory',
                    help="directory name which includes trained model",
                    default="dense_0123",
                    type=str,
                    required=True)

parser.add_argument('--model',
                    help="model architecture",
                    default="densenet121",
                    type=str,
                    required=False)
parser.add_argument('--mixup', action='store_true', default=False,
                help='whether to use manifold mixup or not')
parser.add_argument('--se', action='store_true', default=False,
                help='whether to use Squeeze and excitation block')
parser.add_argument('--shake', action='store_true', default=False,
                help='whether to use shake-shake regularization in network')
parser.add_argument('--tta', action='store_true', default=False,
                help='whether to use Test Time Augmentation')
parser.add_argument('--big', action='store_true', default=False,
                help='whether to use Big Image')

args = parser.parse_args()

assert args.model in __all__, "The model {} doesn't exist in model.py".format(args.model)



BATCH_SIZE = args.batchsize
models = os.listdir(os.path.join("model", args.directory))

min_loss = 1000000000000
# search model for checkpoint having lowest loss.
for name in models:
    if not name[-6:] == ".model":
        continue
    loss = float(name.split("_")[0][7:])
    if loss < min_loss:
        model_name = name
        min_loss = loss

print("lowest loss model name:", model_name)

gpu_name = "cuda:" + str(args.gpu)
device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")
print("device:{}".format(device))

path_test = pd.read_pickle("save/path_test.pkl")
if args.big:
    X_test = pd.read_pickle("save/X_test_399.pkl")
else:
    X_test = pd.read_pickle("save/X_test_300.pkl")
Y_test = pd.read_pickle("save/Y_test.pkl")


print("The number of test images:{}".format(len(Y_test)))
c = Counter(Y_test)
print("test:",c)

# These values are same as the training ones
ch_mean = [237.78516378, 237.16343756, 237.0501237]
ch_std = [146.47616225347468, 146.6169214951974, 145.59586636818233]

test = Dataset(X_test, Y_test, paths=path_test, transform=transforms.Compose([
                                                                              Regularizer(ch_mean, ch_std=ch_std),
                                                                              ToTensor()]))
test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

# model selection
if args.model == "resnet18":
    print("SeNet18")
    model = se_resnet18(2, if_mixup=args.mixup, if_shake_shake=args.shake,
                        first_conv_stride=args.first_stride, first_pool=True).to(device)
elif args.model == "resnet152":
    print("SeNet152")
    model = se_resnet152(2, if_mixup=args.mixup, if_shake_shake=args.shake).to(device)
elif args.model == "densenet121":
    print("DenseNet121")
    model = densenet121(if_mixup=args.mixup, if_selayer=args.se, first_conv_stride=args.first_stride, first_pool=True).to(device)
elif args.model == "densenet169":
    print("DenseNet169")
    model = densenet169(if_mixup=args.mixup, if_selayer=args.se).to(device)
elif args.model == "densenet201":
    print("DenseNet201")
    model = densenet201(if_mixup=args.mixup, if_selayer=args.se).to(device)
elif args.model == "dpn92":
    print("DPN92")
    model = dpn92(num_classes=2, if_selayer=args.se).to(device)
elif args.model == "dpn98":
    print("DPN98")
    model = dpn98(num_classes=2, if_selayer=args.se).to(device)
elif args.model == "dpn131":
    print("DPN131")
    model = dpn131(num_classes=2, if_selayer=args.se).to(device)


MODEL_PATH = os.path.join("model", args.directory, model_name)
checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("==>" + "model loaded:", MODEL_PATH)

# train_loss = pd.read_pickle("save/{}_train_losses.pkl".format(args.directory))
# print("TRAINED EPOCHS: ", len(train_loss))

with torch.no_grad():
    model.eval()

    correct = 0
    y_score = []
    y_pred = []
    img_path = []
    y = []
    i = 0

    for idx, data in enumerate(tqdm(test_iter)):
        path, inputs, labels = data
        img_path.extend(path)
        labels.to(device)

        # Test time augmentation
        if args.tta:
            minibatch_inputs = []
            for input_img_tensor, label in zip(inputs, labels):
                augmented_imgs = TTA_for_img(input_img_tensor)
                augmented_imgs = augmented_imgs
                minibatch_inputs.append(augmented_imgs)

            minibatch_inputs = torch.cat(minibatch_inputs, 0).to(device)

            outputs = model(minibatch_inputs)
            # averaged_pred = torch.mean(preds, 0)
            # max_probability, idx = preds[:, 1].max(dim=0)
            # max_index = torch.argmax(preds[idx])
            #
            # if max_index.cpu().numpy() == label.cpu().numpy():
            #     correct += 1

            y_score.extend(outputs.cpu().numpy()[:, 1])
            # y_pred.append(max_index.cpu().numpy())

        else:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            max_index = outputs.max(dim = 1)[1]
            correct += (max_index == labels).sum().item()

            y_score.extend(outputs.cpu().numpy()[:, 1])

            y_pred.extend(max_index.cpu().numpy())
        y.extend(labels.cpu().numpy())


if args.tta:
    assert len(y_score) % 4 == 0
    for idx, score in enumerate(y_score):
        if idx % 4 == 0:
            augmented_predictions = y_score[idx : idx+4]
            rotated_270 = augmented_predictions[0]
            rotated_180 = augmented_predictions[1]
            rotated_90 = augmented_predictions[2]
            rotated_0 = augmented_predictions[3]



acc = correct/len(X_test)
conf = confusion_matrix(y, y_pred)
tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

auc = roc_auc_score(y, y_score)
fpr, tpr, thresholds = metrics.roc_curve(y, y_score)

get_roc_auc_curve(fpr, tpr, auc)
get_precision_recall_curve(y, y_score)
print("Accuracy:{}\nAUC:{}".format(acc, auc))
print("Confusion matrix:\n", conf)

def get_precision_at_recall_k(k, score, true_label):
    """
    return precision at fixed recall=k
    Args :
        k : recall value
        score : list of 0-1 scores for class1
        true_label : list of 0 or 1 binary labels
    """
    def sigmoid(x):
      return 1 / (1+np.exp(-x))

    score_cp = copy.copy(score)
    true_label_cp = copy.copy(true_label)

    union = list(zip(score_cp, true_label_cp))
    # sort by prediction score
    union.sort(reverse=False)
    sorted_score, sorted_label = zip(*union)

    for idx, threshold in enumerate(sorted_score):
        predicted_as_positive_arr = np.array(sorted_label[idx:])
        predicted_as_negative_arr = np.array(sorted_label[:idx])

        true_positive = np.sum(predicted_as_positive_arr)
        false_positive = len(predicted_as_positive_arr) - true_positive
        false_negative = np.sum(predicted_as_negative_arr)
        true_negative = len(predicted_as_negative_arr) - false_negative

        recall = true_positive / (true_positive+false_negative)
        precision = true_positive / (true_positive+false_positive)

        if recall >= k:
            recall_ = recall
            precision_ = precision
            threshold_ = sigmoid(threshold)
            continue
        else:
            return precision_, recall_, threshold_
k=0.95
precision, recall, threshold = get_precision_at_recall_k(k, y_score, y)
print("precision:recall {} : {:.3f}, recall : {:.3f}, threshold : {:.3f}".format(k, precision, recall, threshold))
k=0.99
precision, recall, threshold = get_precision_at_recall_k(k, y_score, y)
print("precision:recall {} : {:.3f}, recall : {:.3f}, threshold : {:.3f}".format(k, precision, recall, threshold))
k=0.999
precision, recall, threshold = get_precision_at_recall_k(k, y_score, y)
print("precision:recall {} : {:.3f}, recall : {:.3f}, threshold : {:.3f}".format(k, precision, recall, threshold))

# Miss Classified images
fn_path, fp_path = [], []
fn_prob, fp_prob = [], []
for path, pred, prob, true in zip(img_path, y_pred, y_score, y):
    if true == 1 and pred == 1:
        continue
    elif true == 0 and pred == 0:
        # true negative
        continue
    elif true == 1 and pred == 0:
        # false negative
        fn_path.append(path)
        fn_prob.append(prob)
    elif true == 0 and pred == 1:
        # false positive
        fp_path.append(path)
        fp_prob.append(prob)


fp = list(zip(fp_prob, fp_path))
fn = list(zip(fn_prob, fn_path))

fp.sort(reverse=True)
fn.sort(reverse=False)

fp_sorted_prob, fp_sorted_path = zip(*fp)
fn_sorted_prob, fn_sorted_path = zip(*fn)

fp_img_dir = "result/miss_imgs/false_positive/"
fn_img_dir = "result/miss_imgs/false_negative/"

if os.path.exists(fp_img_dir):
    shutil.rmtree(fp_img_dir)
os.makedirs(fp_img_dir)
if os.path.exists(fn_img_dir):
    shutil.rmtree(fn_img_dir)
os.makedirs(fn_img_dir)


# Save miss classified images.
def sigmoid(x):
  return 1 / (1+np.exp(-x))

for path, p in zip(fp_sorted_path, fp_sorted_prob):
    img = Image.open(path)
    img_name = path.split("/")[-1]
    folder = path.split("/")[-3]
    img.save(os.path.join(fp_img_dir, "{:.2f}_{}_{}".format(sigmoid(p), folder, img_name)))

print("="*100)

for path, p in zip(fn_sorted_path, fn_sorted_prob):
    img = Image.open(path)
    img_name = path.split("/")[-1]
    folder = path.split("/")[-3]
    img.save(os.path.join(fn_img_dir, "{:.2f}_{}_{}".format(sigmoid(p), folder, img_name)))



#end of code
