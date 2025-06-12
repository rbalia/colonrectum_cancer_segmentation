from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import transform, exposure, img_as_float
from skimage.util import random_noise
import numpy as np
import cv2 as cv
from skimage import io
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from volumentations import (Compose, Rotate, RandomCropFromBorders, ElasticTransform, Resize, Flip, RandomRotate90,
                            RandomGamma, Transpose)

def image_augmentation(img, msk, prob=1.0):
    img_aug = np.copy(img)
    msk_aug = np.copy(msk)

    def get_augmentation(patch_size):
        return Compose([
            Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            RandomCropFromBorders(crop_value=0.1, p=0.5),
            ElasticTransform((0, 0.2), interpolation=1, p=0.3),
            Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=0.5),
            Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            RandomRotate90((1, 2), p=0.5),
            # GaussianNoise(var_limit=(0, 1), p=0.9),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            Transpose(axes=(1, 2, 0,3), p=0.4),
            Transpose(axes=(2, 0, 1,3), p=0.4),
        ], p=prob)

    aug = get_augmentation(img.shape)

    # with mask
    data = {'image': img_aug, 'mask': msk_aug}
    aug_data = aug(**data)
    img_aug, msk_aug = aug_data['image'], aug_data['mask']

    return img_aug, msk_aug

def image_augmentation_single(img, prob=1.0):
    img_aug = np.copy(img)

    def get_augmentation(patch_size):
        return Compose([
            Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            RandomCropFromBorders(crop_value=0.1, p=0.5),
            ElasticTransform((0, 0.2), interpolation=1, p=0.3),
            Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=0.5),
            Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            RandomRotate90((1, 2), p=0.5),
            # GaussianNoise(var_limit=(0, 1), p=0.9),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            Transpose(axes=(1, 2, 0,3), p=0.4),
            Transpose(axes=(2, 0, 1,3), p=0.4),
        ], p=prob)

    aug = get_augmentation(img.shape)

    # with mask
    data = {'image': img_aug}
    aug_data = aug(**data)
    img_aug = aug_data['image']

    return img_aug

def image_augmentation_noTranspose(img, msk):
    img_aug = np.copy(img)
    msk_aug = np.copy(msk)

    def get_augmentation(patch_size):
        return Compose([
            Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
            RandomCropFromBorders(crop_value=0.1, p=0.5),
            ElasticTransform((0, 0.2), interpolation=1, p=0.3),
            Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
            #Flip(0, p=0.5),
            Flip(1, p=0.5),
            Flip(2, p=0.5),
            RandomRotate90((1, 2), p=0.5),
            # GaussianNoise(var_limit=(0, 1), p=0.9),
            RandomGamma(gamma_limit=(80, 120), p=0.5),
            Transpose(axes=(0, 2, 1, 3), p=0.4),
        ], p=1.0)

    aug = get_augmentation(img.shape)

    # with mask
    data = {'image': img_aug}
    aug_data = aug(**data)
    img_aug = aug_data['image']

    return img_aug, msk_aug

def augmentSetGenerator(img_set, msk_set, n_augmentations, maxCount=0):

    if n_augmentations == 0:
        return img_set, msk_set

    if maxCount > 0:
        total = img_set.shape[0] + maxCount
    else:
        total = img_set.shape[0] * (n_augmentations + 1)

    imgAug_set = np.ndarray((total, *img_set.shape[1:]), dtype=np.float32)
    mskAug_set = np.ndarray((total, *msk_set.shape[1:]), dtype=np.float32)

    i = 0
    j = 0
    # for img, mask in zip(imgs, imgs_mask):
    while i <= total:
        if i >= total:
            break

        imgAug_set[i] = img_set[j]
        mskAug_set[i] = msk_set[j]
        i += 1
        for variant in range(n_augmentations):

            if i >= total:
                break
            imgAug, mskAug = image_augmentation(img_set[j], msk_set[j])
            # If img is empty, retry
            # TODO: Gestire casi anomali
            while np.count_nonzero(imgAug) == 0:
                imgAug, mskAug = image_augmentation(img_set[j], msk_set[j])
            imgAug_set[i] = np.array([imgAug])
            mskAug_set[i] = np.array([mskAug])
            i += 1

            if i >= total:
                break

        j += 1

    return imgAug_set, mskAug_set

def augmentSetGenerator_noTranspose(img_set, msk_set, n_augmentations, maxCount=0):

    if n_augmentations == 0:
        return img_set, msk_set

    if maxCount > 0:
        total = img_set.shape[0] + maxCount
    else:
        total = img_set.shape[0] * (n_augmentations + 1)

    imgAug_set = np.ndarray((total, *img_set.shape[1:]), dtype=np.float32)
    mskAug_set = np.ndarray((total, *msk_set.shape[1:]), dtype=np.float32)

    i = 0
    j = 0
    # for img, mask in zip(imgs, imgs_mask):
    while i <= total:
        if i >= total:
            break

        imgAug_set[i] = img_set[j]
        mskAug_set[i] = msk_set[j]
        i += 1
        for variant in range(n_augmentations):

            if i >= total:
                break
            imgAug, mskAug = image_augmentation_noTranspose(img_set[j], msk_set[j])
            # If img is empty, retry
            # TODO: Gestire casi anomali
            while np.count_nonzero(imgAug) == 0:
                imgAug, mskAug = image_augmentation_noTranspose(img_set[j], msk_set[j])
            imgAug_set[i] = np.array([imgAug])
            mskAug_set[i] = np.array([mskAug])
            i += 1

            if i >= total:
                break

        j += 1

    return imgAug_set, mskAug_set

