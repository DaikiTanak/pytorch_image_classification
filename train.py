import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter, ImageDraw, ImageChops
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from collections import defaultdict
import copy
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import os.path

import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils import data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Sampler

from collections import Counter
from datetime import datetime, timezone, timedelta

from model import SimpleNet, se_resnet18, se_resnet50, se_resnet101, densenet121, densenet169, densenet201, dpn92, dpn131, dpn98, __all__
from dataset import *
from utils import analyze_XML, make_data, load_tairyo_data, load_new_data, analyze_prediction, Train_Logger, line, give_label_to_newdata
import argparse
import random


def sample_lambda_from_beta_distribution(alpha=1.0):
    """
    Return lambda for mixup process.
    Args :
        alpha : A parameter for beta distribution
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


def train(X_train, Y_train, X_val, Y_val, unlabeled_data=None, batch=16, gpu_id=0, epoch=100):
    """
    train model from training and validation data.
    Args :
        X_train : Training data
        Y_train : Labels of training data
        X_val : Validation data
        Y_val : Labels of validation data
        unlabeled_data : Training data which don't have labels (This is for semisupervised learning.)
        batch : batchsize
        gpu_id : GPU id where model is trained at
    """

    assert len(X_train) == len(Y_train), "training data and its labels must have the same length : {}, {}".format(len(X_train), len(Y_train))
    assert len(X_val) == len(Y_val), "validation data and its labels must have the same length : {}, {}".format(len(X_val), len(Y_val))
    if unlabeled_data is not None:
        assert X_train.shape[1:] == X_val.shape[1:] == unlabeled_data.shape[1:], "All data must have the same shape"


    H,W = X_train[0].shape[:2]
    print("img shape: {}*{}".format(H,W))

    BATCH_SIZE = batch
    EPOCH = epoch
    gpu_name = "cuda:" + str(gpu_id)
    device = torch.device(gpu_name if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark=True

    print("=============training config=============")
    print("device:{}".format(device))
    print("MODEL:",args.model)
    print("save dir : ",model_save_dir)
    print("batch size:{}".format(BATCH_SIZE))
    if args.mixup:
        print("Manifold-mixup:{}".format(args.mixup))
    if args.shake:
        print("Shake-shake regularization:{}".format(args.shake))
    if args.vat:
        print("VAT regularization:{}".format(args.vat))
    print("========================================")

    # Calculate the average of pixels per channel
    # def avg_each_channel(path):
    #     img = Image.open(path)
    #     if not len(np.array(img)) == 3:
    #         img = img.convert("RGB")
    #     img = np.asarray(img)
    #     ch_mean = np.average(np.average(img, axis=0), axis=0)
    #     return ch_mean # (3, )
    #
    # ch_means_per_image = Parallel(n_jobs=-1, verbose=-1)([delayed(avg_each_channel)(path) for path in path_dev])
    # ch_mean = np.average(ch_means_per_image, axis=0)
    # print("channel mean:{}".format(ch_mean))
    #
    # def std_each_channel(path):
    #     channel_mean = [237.78516378, 237.16343756, 237.0501237]
    #
    #     img = Image.open(path)
    #     if not len(np.array(img)) == 3:
    #         img = img.convert("RGB")
    #     img = np.asarray(img)
    #     R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    #     R_flat = R.flatten()
    #     G_flat = G.flatten()
    #     B_flat = B.flatten()
    #
    #     R_diff = np.sum(np.square(R_flat - channel_mean[0]))
    #     G_diff = np.sum(np.square(G_flat - channel_mean[1]))
    #     B_diff = np.sum(np.square(B_flat - channel_mean[2]))
    #
    #     return (R_diff, G_diff, B_diff)
    # pixels_diff = Parallel(n_jobs=-1, verbose=-1)([delayed(std_each_channel)(path) for path in path_dev])
    # R_all, G_all, B_all = 0, 0, 0
    # for pixel_diff in pixels_diff:
    #     R_all += pixel_diff[0]
    #     G_all += pixel_diff[1]
    #     B_all += pixel_diff[2]
    #
    # R_std = np.sqrt(R_all/(H*W*len(path_dev)))
    # G_std = np.sqrt(G_all/(H*W*len(path_dev)))
    # B_std = np.sqrt(B_all/(H*W*len(path_dev)))
    #
    # ch_std = [R_std, G_std, B_std]
    # print("channel std:", ch_std)

    ch_mean = [237.78516378, 237.16343756, 237.0501237]
    ch_std = [146.47616225347468, 146.6169214951974, 145.59586636818233]

    c_train = Counter(Y_train)
    c_val = Counter(Y_val)
    c_test = Counter(Y_test)
    print("train:{}, val:{}, test:{}".format(c_train, c_val, c_test))
    print("train data length:{}, validation data length:{}, test data length:{}".format(len(X_train), len(X_val), len(X_test)))


    if args.semisupervised:
        train = MeanTeacherDataset(X_train, Y_train, transform=transforms.Compose([
                                                                                    #ColorJitter(brightness=0.0, contrast=0.4, hue=0.0),
                                                                                    RandomRotate(hard_rotate=False, angle=5),
                                                                                    Regularizer(ch_mean, ch_std=ch_std),
                                                                                    #Samplewise_Regularizer(),
                                                                                    ToTensor()]))

    else:
        train = Dataset(X_train, Y_train, transform=transforms.Compose([#UnsharpMasker(radius=5.0),
                                                                        #ColorJitter(brightness=0.0, contrast=0.4, hue=0.0),
                                                                        #RandomScalor(scale_range=(301, 330), crop_size=H),
                                                                        RandomRotate(hard_rotate=False, angle=5),
                                                                        Regularizer(ch_mean, ch_std=ch_std),
                                                                        #Samplewise_Regularizer(),
                                                                        ToTensor()]))
    val = Dataset(X_val, Y_val, transform=transforms.Compose([#UnsharpMasker(radius=5.0),
                                                            Regularizer(ch_mean, ch_std=ch_std),
                                                            #Samplewise_Regularizer(),
                                                            ToTensor()]))

    dataset_sizes = {"train" : train.__len__(),
                    "val" : val.__len__()}
    print("dataset size:", dataset_sizes)

    val_batch=BATCH_SIZE

    if args.pseudo_label:
        # Class ratio may be unbalance
        class_sample_count = np.array([len(np.where(Y_train == t)[0]) for t in np.unique(Y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in Y_train])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False, sampler=sampler)
    else:
        train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=val_batch, shuffle=False)


    # model selection
    if args.model == "resnet18":
        print("SeNet18")
        model = se_resnet18(2, if_mixup=args.mixup, if_shake_shake=args.shake,
                            first_conv_stride=2, first_pool=True).to(device)
    elif args.model == "resnet50":
        print("SeNet50")
        model = se_resnet50(2, if_mixup=args.mixup, if_shake_shake=args.shake,
                            first_conv_stride=2, first_pool=True).to(device)
    elif args.model == "resnet101":
        print("SeNet101")
        model = se_resnet101(2, if_mixup=args.mixup, if_shake_shake=args.shake,
                            first_conv_stride=2, first_pool=True).to(device)
    elif args.model == "resnet152":
        print("SeNet152")
        model = se_resnet152(2, if_mixup=args.mixup, if_shake_shake=args.shake).to(device)
    elif args.model == "densenet121":
        print("DenseNet121")
        model = densenet121(if_mixup=args.mixup, if_selayer=args.se,
                            first_conv_stride=2, first_pool=True, drop_rate=args.drop_rate).to(device)

    elif args.model == "dpn92":
        print("DPN92")
        model = dpn92(num_classes=2, if_selayer=args.se, if_mixup=args.mixup,
                      first_conv_stride=2, first_pool=True).to(device)
    elif args.model == "dpn98":
        print("DPN98")
        model = dpn98(num_classes=2, if_selayer=args.se, if_mixup=args.mixup).to(device)

    else:
        print("WRONG MODEL NAME")
        input("---------Stop-----------")



    if args.semisupervised:
        """ Declare the teacher model """

        def sigmoid_rampup(current, rampup_length):
            """ Exponential rampup from https://arxiv.org/abs/1610.02242 """
            if rampup_length == 0:
                return 1.0
            else:
                current = np.clip(current, 0.0, rampup_length)
                phase = 1.0 - current / rampup_length
                return float(np.exp(-5.0 * phase * phase))

        def get_current_consistency_weight(epoch):
            # Consistency ramp-up from https://arxiv.org/abs/1610.02242
            return args.consistency * sigmoid_rampup(epoch, rampup_length=int(args.epoch/2))

        def update_teacher(student, teacher, alpha, global_step):
            """
            update parameters of the teacher model.
            Args :
                student : Current model to train
                teacher : Current teacher model
                alpha : A parameter of models mixing weights
                global_step : Global step of training
            """
            alpha = min(1 - 1 / (global_step + 1), alpha)
            for teacher_param, param in zip(teacher.parameters(), student.parameters()):
                teacher_param.data.mul_(alpha).add_(1 - alpha, param.data)

        teacher_model = copy.deepcopy(model)
        consistency_criterion = nn.MSELoss()

    print("model preparing...")
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    bce_loss = torch.nn.BCEWithLogitsLoss()
    softmax = torch.nn.Softmax(dim=1)


    #Set optimizer
    init_learning_rate = args.learning_rate
    optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(EPOCH*0.5), int(EPOCH*0.75)], gamma=0.1)

    lowest_loss = 1000000000000

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    PATH = ""
    global_step = 0
    start_epoch = 0
    Logger = Train_Logger()

    if args.resume:
        # resume training from the latest checkpoint

        checkpoint_names = os.listdir(os.path.join("model", args.directory))

        # find a checkpoint having lowest loss
        min_loss = 1000000000000
        for name in checkpoint_names:
            loss = float(name.split("_")[0][7:])
            if loss < min_loss:
                model_name = name
                min_loss = loss

        MODEL_PATH = os.path.join("/home/tanaka301052/tegaki/model", args.directory, model_name)

        # load model and optimizer from the checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        lowest_loss = checkpoint['lowest_loss']
        global_step = checkpoint["global_step"]

        print("==>" + "model loaded:", model_name)
        print("current epoch:", start_epoch)
        print("lowest loss:", lowest_loss)
        print("resuming...")




    for epoch in range(start_epoch, EPOCH, 1):
        scheduler.step()

        # Training Phase
        model.train()
        train_loss = 0
        train_corrects = 0
        loss = 0

        for i, train_data in enumerate(tqdm(train_iter)):
            global_step += 1
            if args.semisupervised:
                # Get inputs for both student model and teacher model
                samples1, samples2 = train_data
                student_inputs, labels = samples1
                teacher_inputs, labels = samples2

                student_inputs = student_inputs.to(device)
                teacher_inputs = teacher_inputs.to(device)
                labels = labels.to(device)

                # forwarding student
                student_outputs = model(student_inputs)
                _, preds = torch.max(student_outputs, 1)

                # forwarding teacher
                teacher_outputs = teacher_model(teacher_inputs).detach()

                # classification loss for student
                classification_loss = loss_function(student_outputs, labels)
                # consistency loss between student and teacher
                consistency_loss = consistency_criterion(softmax(student_outputs), softmax(teacher_outputs))
                # get weight of consistency loss
                consistency_weight = get_current_consistency_weight(epoch)

                # total loss
                loss = classification_loss + consistency_weight*consistency_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if epoch < int(args.epoch/2):
                    # during ramp up.
                    alpha = 0.99
                else:
                    alpha = 0.999
                update_teacher(model, teacher_model, alpha=alpha, global_step=global_step)

                train_loss += classification_loss.item() * student_inputs.size(0)
                train_corrects += (preds == labels).sum().item()

            else:
                inputs, labels = train_data
                inputs = inputs.to(device)
                labels = labels.to(device)
                if not args.mixup:
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    train_corrects += (preds == labels).sum().item()
                elif args.mixup:
                    lam = sample_lambda_from_beta_distribution(alpha=args.mixup_alpha)
                    lam = torch.from_numpy(np.array([lam]).astype('float32')).to(device)
                    output, reweighted_target = model(inputs, lam=lam, target=labels, device=device)

                    loss = bce_loss(output, reweighted_target)

                train_loss += loss.item() * inputs.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        train_loss /= dataset_sizes["train"]
        train_losses.append(train_loss)
        train_acc = train_corrects / dataset_sizes["train"]
        train_accs.append(train_acc)
        print("=====> train loss:{:5f} Acc:{:5f}".format(train_loss, train_acc))

        # Validation Phase
        model.eval()
        val_loss = 0
        loss = 0
        val_corrects = 0
        for i, val_data in enumerate(val_iter):

            inputs, labels = val_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            max_index = outputs.max(dim = 1)[1]

            loss = loss_function(outputs, labels)
            val_corrects += (max_index == labels).sum().item()

            val_loss += loss.item() * inputs.size(0)

        val_loss /= dataset_sizes["val"]
        val_losses.append(val_loss)

        result = {"epoch":epoch,
                  "train_loss":train_loss,
                  "val_loss":val_loss,
                  "lowest_loss":lowest_loss,
                  "val_accuracy":val_corrects/dataset_sizes["val"],
                  }

        val_accs.append(result["val_accuracy"])

        if val_loss < lowest_loss:
            lowest_loss = val_loss
            if epoch >= 0:
                # Prevent saving the model in the first epoch.

                # remove latest saved model
                if os.path.isfile(PATH):
                    os.remove(PATH)

                model_name = "valloss{:.5f}_epoch{}.model".format(val_loss, epoch)
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)
                PATH = os.path.join(model_save_dir, model_name)
                checkpoint = {"epoch":epoch,
                              "global_step":global_step,
                              "model_state_dict":model.state_dict(),
                              "optim_state_dict":optimizer.state_dict(),
                              "lowest_loss":val_loss}

                torch.save(checkpoint, PATH)
                line("Project:{} \nEPOCH:{} train loss:{:4f}, val Accuracy:{:4f}, val loss:{:4f}, best loss:{:4f}".format(args.directory,
                                                                                                                          epoch,
                                                                                                                          train_loss,
                                                                                                                          result["val_accuracy"],
                                                                                                                          result["val_loss"],
                                                                                                                          lowest_loss))


        print("EPOCH:{} val Accuracy:{:4f}, val loss:{:4f}, best loss:{:4f}".format(epoch, result["val_accuracy"], result["val_loss"], lowest_loss))
        Logger.send_loss_img(str(args.directory), train_losses, val_losses, process_name=args.directory)
        Logger.send_accuracy_img(str(args.directory), train_accs, val_accs, process_name=args.directory)

        output_eval_file = os.path.join("result", "{}_training_log.txt".format(args.directory))
        Logger.write_log(result, output_eval_file)

        Logger.save_history(train_losses, "save/{}_train_losses.pkl".format(args.directory))
        Logger.save_history(val_losses, "save/{}_val_losses.pkl".format(args.directory))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                prog='train.py',
                usage='Training Classifier',
                description='description',
                epilog='end',
                add_help=True,
                )
    parser.add_argument('-b', '--batchsize',
                        help="num of data in each mini-batch",
                        default=16,
                        type=int,
                        required=False)
    parser.add_argument('-g', '--gpu',
                        help="gpu id to run the model at.",
                        default=0,
                        type=int,
                        required=False)

    parser.add_argument('-e', '--epoch',
                        help="num of epoch in training",
                        default=100,
                        type=int,
                        required=False)
    parser.add_argument('--consistency',
                        help="weight of consistency loss",
                        default=50,
                        type=int,
                        required=False)
    parser.add_argument('--folder_num',
                        help="the number of folders to use as additional training data",
                        default=0,
                        type=int,
                        required=False)

    parser.add_argument('--directory',
                        help="model save director name",
                        default="senet18_0110",
                        type=str,
                        required=False)

    parser.add_argument('--model',
                        help="model architecture",
                        default="densenet121",
                        type=str,
                        required=False)

    parser.add_argument('--mixup_alpha',
                        help="alpha in manifold mixup",
                        default=0.2,
                        type=float,
                        required=False)

    parser.add_argument('--learning_rate',
                        help="initial learning rate",
                        default=0.01,
                        type=float,
                        required=False)
    parser.add_argument('--drop_rate',
                        help="drop out rate",
                        default=0.0,
                        type=float,
                        required=False)

    parser.add_argument('--semisupervised',
                        action='store_true', default=False,
                        help="if to treat non tegaki as unlabeled data.",
                        required=False)
    parser.add_argument('--pseudo_label',
                        action='store_true', default=False,
                        help="if to treat non tegaki by pseudo_labeling.",
                        required=False)
    parser.add_argument('--mixup', action='store_true', default=False,
                    help='whether to use manifold mixup or not')
    parser.add_argument('--se', action='store_true', default=False,
                    help='whether to use Squeeze and excitation block')
    parser.add_argument('--shake', action='store_true', default=False,
                    help='whether to use shake-shake regularization in network')
    parser.add_argument('--big_img', action='store_true', default=False,
                    help='whether to use bigger size of images')
    parser.add_argument('--resume', action='store_true', default=False,
                    help='whether to resume training')
    parser.add_argument('--partial_train', action='store_true', default=False,
                    help='whether to train by partial training data')
    parser.add_argument('--train_data_num',
                        help="the number of data to use as training data",
                        default=1000,
                        type=int,
                        required=False)

    args = parser.parse_args()

    assert args.model in __all__, "The model {} doesn't exist in model.py".format(args.model)


    # config
    xml_dir = "data/xml"
    image_dir = "data/sansan_images/GeesImages"
    model_save_dir = os.path.join("model", args.directory)

    if args.big_img:
        img_size=399
        print("======load big data======")
        img = pd.read_pickle("save/img_399.data")
    else:
        img_size=300
        img = pd.read_pickle("save/img_300.data")
    path = pd.read_pickle("save/path.data")
    label = pd.read_pickle("save/label.data")


    print("labeled:{}".format(len(path)))

    path_dev, path_test, X_dev, X_test, Y_dev, Y_test = train_test_split(path, img, label, test_size=0.5, random_state=0)
    pd.to_pickle(path_test, "save/path_test.pkl")
    pd.to_pickle(Y_test, "save/Y_test.pkl")
    if args.big_img:
        pd.to_pickle(X_test, "save/X_test_399.pkl")
    else:
        pd.to_pickle(X_test, "save/X_test_300.pkl")

    # devide development data into train data and validation data.
    X_train, X_val, Y_train, Y_val = train_test_split(X_dev, Y_dev, test_size=0.5, random_state=0)
    train_counter = Counter(Y_train)
    print(train_counter)


    if args.partial_train:
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        data_num = args.train_data_num
        extracted_idx = np.random.choice(X_train.shape[0], data_num, replace=False)
        X_train = X_train[extracted_idx, :]
        assert X_train.shape[0] == data_num
        X_train = list(X_train)
        Y_train = list(Y_train[extracted_idx])


    train_counter = Counter(Y_train)
    print(train_counter)

    # load new imgs.
    dirs_num = args.folder_num
    new_tegaki, new_nontegaki = load_new_data(img_size=img_size, num_folder=dirs_num)

    print("new tegaki:{}, new nontegaki:{}".format(len(new_tegaki), len(new_nontegaki)))

    if args.pseudo_label:
        X_train = X_train + new_tegaki + new_nontegaki
        give_label_to_newdata(new_nontegaki)
        pseudo_label = analyze_prediction(threshold=0.99)
        label_count = Counter(pseudo_label)
        print("pseudo_label:", label_count)
        Y_train = Y_train + [1]*len(new_tegaki) + pseudo_label
    elif args.semisupervised and not args.pseudo_label:
        """ give -1 label to new nontegaki data."""
        X_train = X_train + new_tegaki + new_nontegaki
        Y_train = Y_train + [1]*len(new_tegaki) + [-1]*len(new_nontegaki)
    else:
        X_train = X_train + new_tegaki + new_nontegaki
        Y_train = Y_train + [1]*len(new_tegaki) + [0]*len(new_nontegaki)

    assert np.array(X_train).shape[1] == np.array(X_train).shape[2]

    line("start training : "+args.directory+"\ngpu : "+str(args.gpu))
    train(X_train, Y_train, X_val, Y_val, batch=args.batchsize, gpu_id=args.gpu, epoch=args.epoch)
    line("finish training : "+args.directory+"\ngpu : "+str(args.gpu))
