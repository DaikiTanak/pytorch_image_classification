from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
from skimage import util
from skimage.util import random_noise
import random
from PIL import Image, ImageFilter, ImageDraw, ImageChops

from scipy.misc import imresize
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class Regularizer(object):
    def __init__(self, ch_mean, ch_std=[255.0, 255.0, 255.0]):
        assert len(ch_mean) == 3 and len(ch_std) == 3

        self.ch_mean = ch_mean
        self.ch_std = ch_std

    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img, label = sample

        img = np.asarray(img, dtype="float64")
        img[:, :, 0] -= self.ch_mean[0]
        img[:, :, 1] -= self.ch_mean[1]
        img[:, :, 2] -= self.ch_mean[2]

        img[:, :, 0] /= self.ch_std[0]
        img[:, :, 1] /= self.ch_std[1]
        img[:, :, 2] /= self.ch_std[2]

        return img, label

class Samplewise_Regularizer(object):
    # Global Contrast Normalization
    def __init__(self):
        pass
    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img, label = sample

        img = np.asarray(img, dtype="float64")

        assert img.shape == (len(img[0]), len(img[0]), 3)
        R = img[:, :, 0].flatten()
        G = img[:, :, 1].flatten()
        B = img[:, :, 2].flatten()

        R_mean = np.mean(R)
        G_mean = np.mean(G)
        B_mean = np.mean(B)

        R_std = np.std(R)
        G_std = np.std(G)
        B_std = np.std(B)

        img[:, :, 0] -= R_mean
        img[:, :, 1] -= G_mean
        img[:, :, 2] -= B_mean

        img[:, :, 0] /= R_std
        img[:, :, 1] /= G_std
        img[:, :, 2] /= B_std

        return img, label


class GaussianNoiser(object):
    def __init__(self, mean=0, variance=0.01):
        self.mean = mean
        self.variance = variance

    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """

        img_arr, label = sample
        noised_img = random_noise(img_arr, mode='gaussian',
                                  mean=self.mean, var=self.variance)*255

        return (noised_img, label)


class ColorJitter(object):
    def __init__(self, brightness=0.3, contrast=0.0, hue=0.0):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue

    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img_arr, label = sample
        PIL_img = Image.fromarray(np.uint8(img_arr))

        # Only change brightness
        jitter = torchvision.transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=0.0, hue=self.hue)
        transformed_PIL_img = jitter(PIL_img)
        transformed_img_arr = np.array(transformed_PIL_img, dtype="uint8")
        return (transformed_img_arr, label)

class UnsharpMasker(object):
    def __init__(self, radius=5.0):
        self.radius = radius
    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img_arr, label = sample
        PIL_img = Image.fromarray(np.uint8(img_arr))

        img_unsharped = PIL_img.filter(ImageFilter.UnsharpMask(radius=self.radius, percent=130))
        unsharped_img_arr = np.array(img_unsharped, dtype="uint8")
        return (unsharped_img_arr, label)

class RandomScalor(object):
    def __init__(self, scale_range=(301, 330), crop_size=300):
        self.scale_range = scale_range
        self.crop_size = crop_size


    def scale_augmentation(self, image):
        def random_crop(image, crop_size=(290, 290)):
            """
                image : np.array
            """
            h, w, _ = image.shape

            top = np.random.randint(0, h - crop_size[0])
            left = np.random.randint(0, w - crop_size[1])

            bottom = top + crop_size[0]
            right = left + crop_size[1]

            image = image[top:bottom, left:right, :]
            return image

        scale_size = np.random.randint(self.scale_range[0], self.scale_range[1])
        image = transform.resize(image, (scale_size, scale_size), preserve_range=True)
        image = random_crop(image, crop_size=(self.crop_size, self.crop_size))
        return image


    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img_arr, label = sample
        return (self.scale_augmentation(img_arr), label)




class RandomRotate(object):
    """
        Rotate image by random angle. -3 ~ +3
    """
    def __init__(self, hard_rotate=False, angle=3):
        self.hard_rotate=hard_rotate
        self.angle=angle

    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img, label = sample
        H,W = img.shape[:2]

        def rotate_image(image):
            return transform.rotate(image, angle=random.randint(-self.angle, self.angle), resize=True, center=None, mode="edge", preserve_range=True)
        rt_img = rotate_image(img)
        rt_img = transform.resize(rt_img, (H,W), mode="edge")

        #Then rotate by 90, 180, 270.
        if self.hard_rotate:
            rand = random.random()
            if rand < 0.2:
                rt_img = transform.rotate(rt_img, angle=90)
            elif rand < 0.4:
                rt_img = transform.rotate(rt_img, angle=-90)
            elif rand < 0.5:
                rt_img = transform.rotate(rt_img, angle=180)

        return (rt_img, label)

class ToTensor(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        """
        Args :
            sample : tuple of (image, label). image is numpy array
        """
        img, label = sample
        img = torch.tensor(img, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        # H,W,C ---> C,H,W
        img = img.permute(2,0,1)
        return (img, label)



class Dataset(Dataset):
    """ image dataset """

    def __init__(self, imgs, labels, paths=None, transform=None):
        """
        Args :
            imgs : np.array (n_samples, H, W, channels)
            labels : label of images (n_samples,)
            paths : paths of images
            transform : transformation on images
        """

        self.imgs = imgs
        self.labels = labels
        self.paths = paths
        if paths is not None:
            print("Set up dataloader with path.")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        label = self.labels[idx]

        sample = (image, label)

        if self.transform:
            sample = self.transform(sample)

        if self.paths is not None:
            img, label = sample
            path = self.paths[idx]
            sample_with_path = (path, img, label)
            return sample_with_path

        return sample




class MeanTeacherDataset(Dataset):
    """ image dataset for Mean teachers and students.
        This dataset returns a pair of images made from tha same image.
    """

    def __init__(self, imgs, labels, paths=None, transform=None):
        """
        Args :
            imgs : np.array (n_samples, H, W, channels)
            labels : label of images (n_samples,)
            paths : paths of images
            transform : transformation on images
        """

        print("This is MeanTeacher Dataset")
        self.imgs = imgs
        self.labels = labels
        self.paths = paths
        if paths is not None:
            print("Set up dataloader with path.")
        if transform is not None:
            print("CAUTION : transformer is None! Student and teacher will get the same images")

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        image = self.imgs[idx]
        label = self.labels[idx]

        sample = (image, label)

        if self.transform:
            sample1 = self.transform(sample)
            sample2 = self.transform(sample)
            return (sample1, sample2)

        if self.paths is not None:
            img, label = sample
            path = self.paths[idx]
            sample_with_path = (path, img, label)
            return sample_with_path

        return sample
