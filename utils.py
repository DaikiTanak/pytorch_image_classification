import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torchvision import transforms, utils
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
import gc
import pandas as pd
import os
from PIL import Image, ImageFilter, ImageDraw, ImageChops
import xml.etree.ElementTree as ET
from collections import defaultdict
import copy
import cv2
from joblib import Parallel, delayed

from torch.autograd import Variable
import errno
from tqdm import tqdm
import requests
import shutil

from collections import Counter
from datetime import datetime, timezone, timedelta
import random

from model import SimpleNet, se_resnet18, se_resnet152, densenet121, densenet169, densenet201, dpn92, dpn131, dpn98
from dataset import *

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import skimage

import argparse

from collections import Counter



class Train_Logger(object):
    """
    Logger for training
    """
    def __init__(self):
        self.line_token = "enter your line token"

    def write_log(self, result, output_eval_file):
        """
        save log
        Args :
            result : result dictionary, ex) key:"loss", value:0.0014
            output_eval_file : path of log file
        """
        with open(output_eval_file, "a") as writer:
            for key, value in result.items():
                writer.write("%s = %s\n" % (key, str(value)))
            writer.write("\n")

    def save_history(self, values, path):
        """
        save list of values (loss or accuracy, for example) as pickle file
        Args :
            values : list of values to save
            path : pickle file path
        """
        pd.to_pickle(values, path)

    def send_loss_img(self, message, train_losses, val_losses, process_name=""):
        """
        send loss and accuracy history via LINE api.
        Args :
            message : message to send
            train_losses : training loss history
            val_losses : validation loss history
            process_name : process name to show in image
        """
        assert len(train_losses) == len(val_losses), "losses must have the same length"
        PATH = "result/loss_{}.jpg".format(process_name)
        if os.path.isfile(PATH):
            os.remove(PATH)

        epoch = len(train_losses)
        fig = plt.figure()
        x = np.linspace(1, epoch, epoch)
        plt.plot(x, train_losses, label="train loss")
        plt.plot(x, val_losses, label="val loss")
        plt.title(process_name)
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.xticks(np.arange(1, epoch+1.0, 10.0))
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(PATH)

        line_token = self.line_token
        line_notify_token = line_token
        line_notify_api = 'https://notify-api.line.me/api/notify'
        message = message
        payload = {'message': message}
        files = {"imageFile" : open(PATH, "rb")}
        headers = {'Authorization': 'Bearer ' + line_notify_token}
        try:
            line_notify = requests.post(line_notify_api, headers=headers, files=files, data=payload)
        except requests.exceptions.ConnectionError:
            print("line notify connection error")



    def send_accuracy_img(self, message, train_acc, val_acc, process_name=""):
        assert len(train_acc) == len(val_acc)
        PATH = "result/acc_{}.jpg".format(process_name)
        if os.path.isfile(PATH):
            os.remove(PATH)

        epoch = len(train_acc)
        fig = plt.figure()
        x = np.linspace(1, epoch, epoch)
        plt.plot(x, train_acc, label="train Accuracy")
        plt.plot(x, val_acc, label="val Accuracy")
        plt.title(process_name)
        plt.xlabel('EPOCH')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(1, epoch+1.0, 10.0))
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(PATH)

        line_token = self.line_token
        line_notify_token = line_token
        line_notify_api = 'https://notify-api.line.me/api/notify'
        message = message
        payload = {'message': message}
        files = {"imageFile" : open(PATH, "rb")}
        headers = {'Authorization': 'Bearer ' + line_notify_token}
        try:
            line_notify = requests.post(line_notify_api, headers=headers, files=files, data=payload)
        except requests.exceptions.ConnectionError:
            print("line notify connection error")

def line(Me):
    """
    Send message via LINE Notify api
    """
    line_token = "enter your line token"
    line_notify_token = line_token
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = '\n' + Me
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    try:
        line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    except requests.exceptions.ConnectionError:
        print("line notify connection error")

def learning_history(train_losses, val_losses):

    epoch = len(train_losses)
    fig = plt.figure()
    x = np.linspace(1, epoch, epoch)
    plt.plot(x, train_losses, label="train loss")
    plt.plot(x, val_losses, label="val loss")
    plt.title('Learning history')
    plt.xlabel('EPOCH')
    plt.ylabel('LOSS')
    plt.xticks(np.arange(1, epoch+1.0, 10.0))
    plt.ylim([0, 1])
    plt.legend()
    plt.savefig("result/loss.jpg")
    plt.show()

    val_lowest_loss = np.min(val_losses)
    print("validation lowest loss:", val_lowest_loss)



def get_roc_auc_curve(fpr, tpr, auc):
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %.5f)'%auc)
    plt.legend()
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.savefig("result/roc_auc.jpg")
    return 0

def get_precision_recall_curve(y, y_score):
    precision, recall, _ = precision_recall_curve(y, y_score)
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    fig = plt.figure()
    plt.step(recall, precision, color='red', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='red', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.savefig("result/precision_recall_curve.jpg")

def analyze_XML(xml_dir):

    # XMLアノテーションの解析
    date_list = sorted(os.listdir(xml_dir))

    # Key : Name of image
    # Value : The box of handwritten letters
    annotation_dic = defaultdict(list)
    all_count = 0
    not_annotated = 0
    annotated = 0
    for date in date_list:
        if date == ".DS_Store":
            continue
        result_files = os.listdir(os.path.join(xml_dir, date))
        print(result_files)

        for result in result_files:
            path = os.path.join(xml_dir, date, result)
    #         print("PATH", path)

            #Reading XML files.
            tree = ET.ElementTree(file=path)
            # treeのroot要素を取得
            root = tree.getroot()
            # XMLの構造体の中に順に潜って要素を取得
            for fruits in root:
                # <ImageDirectory>, <Images>
                for image_sample in fruits:
                    #<ImageSample>
                    location_list = []
                    for item in image_sample:
                        #Rects, FileName
                        if item.tag == "FileName":
    #                         print("FileName:", item.text)
                            filename = item.text
                            all_count += 1

                        elif item.tag == "Rects" and item.text is not None:
                            # アノテーションが存在.
                            annotated +=1
                            for rect_item in item:
                                #<Rectangle>
                                each_location_list = [] #[X,Y,W,H]
                                for loc_item in rect_item:
                                    #<Location>, <Size>
                                    if loc_item.tag == "Location":
                                        for dim in loc_item:
    #                                         print("Location:", dim.tag, dim.text)
                                            each_location_list.append(dim.text)
                                    elif loc_item.tag == "Size":
                                        for dim in loc_item:
    #                                         print("Size:", dim.tag, dim.text)
                                            each_location_list.append(dim.text)
                                location_list.append(each_location_list)
                        else:
                            # item.tag == Rects and item.text is None.
                            not_annotated += 1
                            assert item.text is None

                    if len(annotation_dic[filename]) > 0 and len(location_list) == 0:
                        #print("already exists")
                        continue
                    assert not (len(annotation_dic[filename]) > 0 and len(location_list) > 0)
                    annotation_dic[filename] = location_list


    annotation_dic_copy = copy.deepcopy(annotation_dic)
    for k,v in annotation_dic_copy.items():
        if len(v) > 0 :
            pass
            # print(k,v)
        elif len(v) == 0:
            del(annotation_dic[k])
    del annotation_dic_copy

    print("anno num", len(annotation_dic))
    return annotation_dic


def give_label_to_newdata(new_nontegaki):
    """
    Pseudo labeling using pretrained model
    Args :
        new_nontegaki : data to give pseudo labels
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = densenet121(if_mixup=True, if_selayer=True,
    #                     first_conv_stride=2, first_pool=True, drop_rate=0).to(device)
    model = se_resnet18(2, if_mixup=False, if_shake_shake=False,
                        first_conv_stride=2, first_pool=True).to(device)

    MODEL_PATH = "model/senet18_0130_f/valloss0.12875_epoch41.model"
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    print("==>" + "model loaded:", MODEL_PATH)

    # new_tegaki, new_nontegaki = load_new_data(img_size=300, num_folder=num_folder)

    ch_mean = [237.78516378, 237.16343756, 237.0501237]
    ch_std = [146.47616225347468, 146.6169214951974, 145.59586636818233]
    test = Dataset(new_nontegaki, [0]*len(new_nontegaki), transform=transforms.Compose([#UnsharpMasker(radius=5.0),
                                                                                      Regularizer(ch_mean, ch_std=ch_std),
                                                                                      #Samplewise_Regularizer(),
                                                                                      ToTensor()]))
    test_iter = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        model.eval()

        correct = 0
        y_score = []
        y_pred = []
        img_path = []
        y = []
        i = 0

        for idx, data in enumerate(tqdm(test_iter)):
            inputs, labels = data

            inputs = inputs.to(device)

            outputs = model(inputs)
            max_index = outputs.max(dim = 1)[1]

            y_score.extend(softmax(outputs).cpu().numpy()[:, 1])
    model.cpu()
    del model
    pd.to_pickle(y_score, "save/prediction_newdata.pkl")

    return y_score

def analyze_prediction(threshold=0.999):
    """
    Decide pseudo labels by given threshold
    Args :
        threshold : if score < threshold, give label 0. Otherwise give label 0.
    """
    print("threshold:", threshold)
    y_pred = pd.read_pickle("save/prediction_newdata.pkl")
    print("total length:", len(y_pred))

    labels = []
    for pred in y_pred:
        if pred < 1-threshold:
            labels.append(0)
        elif pred > threshold:
            labels.append(1)
        else:
            labels.append(-1)
    labels_count = Counter(labels)
    print("pseudo label result:{}".format(labels_count))

    return labels

def make_data(annotation_dic=None, resize_size=300,
              tegaki_dir="data/handwritten_data/GeesImages",
              non_tegaki_dir="data/no_handwritten_data_2/GeesImages",
              sansan_public_images_dir="data/open_images/images"):

    # print("miss labeled:{}, noise:{}, flipped:{}".format(len(miss_labeled), len(noise_images), len(flipped_images)))

    # date_list = sorted(os.listdir(image_dir))
    tegaki_images_names = os.listdir(tegaki_dir)
    tegaki_images_paths = [os.path.join(tegaki_dir, name) for name in tegaki_images_names]
    non_tegaki_images_names = os.listdir(non_tegaki_dir)
    non_tegaki_images_paths = [os.path.join(non_tegaki_dir, name) for name in non_tegaki_images_names]


    # there are tegaki images in non_tegaki dir.
    f = open("data/no_handwritten_data_2/include_tegaki.txt")
    not_handwritten_miss_labeled = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    # remove \n

    not_handwritten_miss_labeled = [s.strip() for s in not_handwritten_miss_labeled]


    f = open("data/handwritten_data/miss_label.txt")
    handwritten_miss_labeled = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    # remove \n
    handwritten_miss_labeled = [s.strip() for s in not_handwritten_miss_labeled]
    print("miss label num : {}".format(len(not_handwritten_miss_labeled)))


    not_handwritten_noise_images = ["13734928222.jpg", "13734931647.jpg", "13734932479.jpg", "13734937619.jpg", "13753642933.jpg",
                                    "13774511422.jpg", "13774597947.jpg", "13775027592.jpg", "13778501866.jpg", "13788943565.jpg",
                                    "13810564552.jpg", "13810566550.jpg", "13810566814.jpg", "13810566817.jpg", "13813865129.jpg",
                                    "13813889833.jpg", "13825504926.jpg", "13826582018.jpg", "13829222658.jpg", "13839159699.jpg",
                                    "13853119223.jpg", "13856327361.jpg", "13856920625.jpg", "13862120304.jpg", "13862185462.jpg",
                                    "13862415528.jpg", "13878854271.jpg", "13878854303.jpg", "13878854576.jpg"]

    not_handwritten_flipped_images = ["13722486225.jpg", "13769967820.jpg", "13813931732.jpg"]

    not_handwritten_left_down = ["13749699857.jpg"]
    not_handwritten_right_down = ["13766568094.jpg", "13796278235.jpg", "13813924480.jpg", "13836110234.jpg", "13841021505.jpg",
                                  "13842240180.jpg", "13867765000.jpg", "13877905805.jpg", "13878879551.jpg", "13880802374.jpg",
                                  "13884711418.jpg", "13885398628.jpg", "13885398919.jpg"]

    img_count = 0
    H, W = [], []

    # The process to be parallelized
    def img_open(path, label, if_non_tegaki):

        img_name = path.split("/")[-1]

        assert path[-4:] == ".jpg" or path[-4:] == ".png"

        img = Image.open(path)
        if not len(np.array(img)) == 3:
            img = img.convert("RGB")
        width = img.width
        height = img.height

        if if_non_tegaki :

            if img_name in not_handwritten_left_down:
                img = img.rotate(270, expand=True)
            elif img_name in not_handwritten_right_down:
                img = img.rotate(90, expand=True)
            elif img_name in not_handwritten_flipped_images:
                img = img.rotate(180, expand=True)

        img_resize_300 = img.resize((300, 300), Image.LANCZOS)
        img_resize_399 = img.resize((399, 399), Image.LANCZOS)

        img_resize_300_array = np.array(img_resize_300, dtype="uint8")
        img_resize_399_array = np.array(img_resize_399, dtype="uint8")

        # modify label
        if if_non_tegaki:
            if img_name in not_handwritten_miss_labeled:
                label = 1
            elif img_name in not_handwritten_noise_images:
                label = -1
            else:
                label = 0
        else:
            if img_name in handwritten_miss_labeled:
                label = 0

        return (path, img_resize_300_array, img_resize_399_array, label)

    # tatenaga = 0
    # yokonaga = 0
    # for path in tegaki_images_paths:
    #     img = Image.open(path)
    #     if not len(np.array(img)) == 3:
    #         img = img.convert("RGB")
    #     width = img.width
    #     height = img.height
    #     if width > height:
    #         yokonaga += 1
    #     else:
    #         tatenaga += 1
    # for path in non_tegaki_images_paths:
    #     img = Image.open(path)
    #     if not len(np.array(img)) == 3:
    #         img = img.convert("RGB")
    #     width = img.width
    #     height = img.height
    #     if width > height:
    #         yokonaga += 1
    #     else:
    #         tatenaga += 1
    # print("横長:", yokonaga)
    # print("縦長:", tatenaga)
    # input("==============")

    tegaki = Parallel(n_jobs=-1, verbose=-1)([delayed(img_open)(path, 1, False) for path in tegaki_images_paths])
    non_tegaki = Parallel(n_jobs=-1, verbose=-1)([delayed(img_open)(path, 0, True) for path in non_tegaki_images_paths])

    data = tegaki + non_tegaki

    img_300_data, label_data = [], []
    img_399_data = []
    path_list = []
    for d in data:
        path, image_300, image_399, label = d
        if label == -1:
            # noise image.
            continue
        path_list.append(path)
        img_300_data.append(image_300)
        img_399_data.append(image_399)
        label_data.append(label)


    print("len of data:", len(img_300_data))

    # list of numpy array.
    return (path_list, img_300_data, img_399_data, label_data)


def noise_img_detector(img):
    """
    Detect noise images whose pixels have alomost same values
    Args :
        img : numpy array of an image

    Return : whether img is noise img or not.
        True : Noise
        False : Normal img
    """
    img_arr = np.asarray(img)

    if len(img_arr.shape) == 3:
        #convert to grayscale img.
        img_arr = skimage.img_as_ubyte(rgb2gray(img_arr))

    pixel_std = np.std(img_arr.flatten())

    if pixel_std < 5.0:
        return True
    else:
        return False



def make_data_(directory_path):
    images_names = os.listdir(directory_path)
    images_paths = [os.path.join(directory_path, name) for name in images_names]

    # The process to be parallelized
    def img_open(path, label):

        img_name = path.split("/")[-1]
        assert path[-4:] == ".jpg" or path[-4:] == ".png" or path[-5:] == ".jpeg"

        img = Image.open(path)

        if noise_img_detector(img):
            label = -1

        if not len(np.array(img)) == 3:
            img = img.convert("RGB")
        width = img.width
        height = img.height

        img_resize_300 = img.resize((300, 300), Image.LANCZOS)
        img_resize_399 = img.resize((399, 399), Image.LANCZOS)

        img_resize_300_array = np.array(img_resize_300, dtype="uint8")
        img_resize_399_array = np.array(img_resize_399, dtype="uint8")


        return (path, img_resize_300_array, img_resize_399_array, label)

    label_ = 1
    data = Parallel(n_jobs=-1, verbose=-1)([delayed(img_open)(path, label_) for path in images_paths])

    path_list, img_300_list, img_399_list, label_list = [], [], [], []
    for d in data:
        path, image_300, image_399, label = d

        if label == -1:
            continue

        path_list.append(path)
        img_300_list.append(image_300)
        img_399_list.append(image_399)
        label_list.append(label)
    return (path_list, img_300_list, img_399_list, label_list)

def to_pickle_newdata():
    """ 画像をnp.arrayにしてpickleしておく """
    folders = ["data/tairyo/bcid"+str(i) for i in range(10)]

    for folder_path in tqdm(folders):
        path, img_300, img_399, label = make_data_(folder_path)
        pd.to_pickle(img_300, "save/{}_300.img".format(folder_path.split("/")[-1]))
        pd.to_pickle(img_399, "save/{}_399.img".format(folder_path.split("/")[-1]))
        pd.to_pickle(path, "save/{}.path".format(folder_path.split("/")[-1]))
        print("num of img:{} noise img:{}".format(len(path), 10000 - len(path)))
        del path, img_300, img_399
        gc.collect()

    folders = ["data/tairyo/no_date_"+str(i) for i in range(10)]
    for folder_path in tqdm(folders):
        path, img_300, img_399, label = make_data_(folder_path)
        pd.to_pickle(img_300, "save/{}_300.img".format(folder_path.split("/")[-1]))
        pd.to_pickle(img_399, "save/{}_399.img".format(folder_path.split("/")[-1]))
        pd.to_pickle(path, "save/{}.path".format(folder_path.split("/")[-1]))
        print("num of img:{} noise img:{}".format(len(path), 10000 - len(path)))
        del path, img_300, img_399
        gc.collect()
    print("done")



def load_tairyo_data(img_size=300, num_folder=10):

    folders = ["bcid"+str(i) for i in range(num_folder)]

    path = []
    img = []
    for folder in tqdm(folders):
        if img_size==399:
            pkl_name = folder + "_399.img"
        else:
            pkl_name = folder + "_300.img"
        save_path = os.path.join("save", pkl_name)
        img_list = pd.read_pickle(save_path)
        path_list = pd.read_pickle(os.path.join("save", folder+".path"))

        path.extend(path_list)
        img.extend(img_list)

    unlabeled_path = []
    unlabeled_img = []

    folders = ["no_date_"+str(i) for i in range(num_folder)]
    for folder in tqdm(folders):
        if img_size==399:
            pkl_name = folder + "_399.img"
        else:
            pkl_name = folder + "_300.img"
        save_path = os.path.join("save", pkl_name)
        img_list = pd.read_pickle(save_path)
        path_list = pd.read_pickle(os.path.join("save", folder+".path"))

        unlabeled_path.extend(path_list)
        unlabeled_img.extend(img_list)

    return (path, img), (unlabeled_path, unlabeled_img)


def load_new_data(img_size=300, num_folder=10):
    print("loading new data...")

    (new_tegaki_path, new_tegaki_img), (new_nontegaki_path, new_nontegaki_img) = load_tairyo_data(img_size=img_size, num_folder=num_folder)
    duplicated = pd.read_pickle("save/duplicated_name.pkl")

    new_tegaki = []
    for p, i in zip(new_tegaki_path, new_tegaki_img):
        name = p.split("/")[-1]
        if name not in duplicated:
            new_tegaki.append(i)

    new_nontegaki = []
    for p, i in zip(new_nontegaki_path, new_nontegaki_img):
        name = p.split("/")[-1]
        if name not in duplicated:
            new_nontegaki.append(i)

    return new_tegaki, new_nontegaki


def clean_new_data():
    """ 被りがある時は除去するため、被っているファイル名を探す """
    path = pd.read_pickle("save/path.data")
    already_have = []
    for p in path:
        name = p.split("/")[-1]
        already_have.append(name)

    (tairyo_path, _, _), (unlabeled_path, _, _) = load_tairyo_data(big=False)
    new_paths = tairyo_path + unlabeled_path
    duplicated = []
    for path in new_paths:
        name = path.split("/")[-1]
        if name in already_have:
            duplicated.append(name)
    pd.to_pickle(duplicated, "save/duplicated_name.pkl")
    return duplicated


if __name__ == "__main__":
    # give_label_to_newdata()

    # (path_list, img_300_list, img_399_list, label_list) = make_data(annotation_dic=None,
    #                                               tegaki_dir="data/handwritten_data/GeesImages",
    #                                               non_tegaki_dir="data/no_handwritten_data_2/GeesImages",
    #                                               sansan_public_images_dir="data/open_images/images")
    #
    # pd.to_pickle(path_list, "save/path.data")
    # pd.to_pickle(img_300_list, "save/img_300.data")
    # pd.to_pickle(img_399_list, "save/img_399.data")
    # pd.to_pickle(label_list, "save/label.data")
    # print("done")

    to_pickle_newdata()
    # give_label_to_newdata()

    # labels = analyze_prediction(threshold=0.99)
    # print("utils.py")

    pass
















    ##
