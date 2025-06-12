from __future__ import print_function

import os
import glob
import re
import fnmatch
import sys

import numpy as np
#import cv2 as cv
import skimage

from skimage.io import imsave, imread, imread_collection
from skimage.transform import resize
from skimage import io
from skimage import img_as_ubyte, img_as_float
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from src import configs as conf, plotter
import matplotlib.pyplot as plt
import tensorflow as tf

def normalize(arr):
    if np.count_nonzero(arr)>0:
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)
    else:
        return arr

def thresholdMask(mask, t):
    mask_th = mask.copy()
    mask_th[mask_th >= t] = 1.
    mask_th[mask_th < t] = 0
    return mask_th

def thresholdMaskSet(maskset_or, t):
    maskset = np.copy(maskset_or)
    i = 0
    for mask in maskset:
        maskset[i] = thresholdMask(mask,t)
        i+=1
    return maskset

def setGray2Color(nparray):
    nparray = nparray[:, :, :, 0]
    nparray = skimage.color.gray2rgb(nparray)
    return nparray

def setColor2Gray(nparray):
    nparray = nparray[:, :, :, 0]
    nparray = np.expand_dims(nparray, axis=-1)
    return nparray

def genROIs_single(img, msk):
    msk_ubyte = img_as_ubyte(msk)

    # Extract bounding box coordinates from mask
    x, y, w, h = cv.boundingRect(msk_ubyte)
    ROI = img[y:y + h, x:x + w]
    ROI_mask = resize(msk, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)

    return ROI, ROI_mask

def genROIs_fromSet(imgs, msks):
    msks_ubyte = img_as_ubyte(msks)
    msks_ubyte = setColor2Gray(msks_ubyte)

    ROIs = []
    ROIs_masks = []
    # Extract bounding box coordinates from mask
    i=0
    for img, msk in zip(imgs, msks_ubyte):
        x, y, w, h = cv.boundingRect(msk)
        if w > 0 and h > 0:
            roi = img[y:y + h, x:x + w,:]
            roi_mask = msk[y:y + h, x:x + w, :]
            ROIs.append(resize(roi, (conf.img_rows, conf.img_cols, imgs.shape[3]), preserve_range=False))
            ROIs_masks.append(resize(roi_mask, (conf.img_rows, conf.img_cols, msks.shape[3]), preserve_range=False))
        else:
            ROIs.append(resize(img, (conf.img_rows, conf.img_cols, imgs.shape[3]), preserve_range=False))
            ROIs_masks.append(resize(msk, (conf.img_rows, conf.img_cols, msks.shape[3]), preserve_range=False))
        i += 1

    #for roi in ROIs:
    #    io.imshow(roi)
    #    plt.show()
    ROIs = np.array(ROIs)
    ROIs_masks = np.array(ROIs_masks)

    return ROIs, ROIs_masks

def combineMasks(path, maskNameList, asGray):
    mask = imread(os.path.join(path, maskNameList[0]), as_gray=asGray)
    if len(maskNameList) > 1:
        for maskName in maskNameList[1:]:
            otherMask = imread(os.path.join(path, maskName), as_gray=asGray)
            mask[otherMask > 0] = 255

    return mask

def saveLabelledROISet(label, dataset, saveMask=True):
    path = os.path.join(conf.setDir, label + '/')

    # Filter out the masks
    images_list = [f for f in os.listdir(path) if re.match('^((?!mask).)*$', f)]
    total = len(images_list)

    # Initialize the numpy arrays
    imgs = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)
    if saveMask:
        msks = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)

    # Initialize the number of channels flag
    asGray = True

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    i = 0
    for fileNameFull in images_list:
        fileName = fileNameFull.split(".")[0]

        if saveMask:
            associated_masks = fnmatch.filter(os.listdir(path), fileName + "_mask*.png")

        # Read image and all associated masks
        img = imread(os.path.join(path, fileNameFull), as_gray=asGray)
        if saveMask:
            msk = combineMasks(path, associated_masks, asGray)

        # Normalization
        img = img_as_float(img / np.max(img))
        if saveMask:
            if np.max(msk) > 0:
                msk = img_as_float(msk / np.max(msk))

        # Extract ROIS
        img, msk = genROIs_single(img, msk)

        # Resize
        img = resize(img, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)
        if saveMask:
            msk = resize(msk, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)

        # Thresholding
        if saveMask:
            msk[msk > 0.5] = 1.0
            msk[msk <= 0.5] = 0.0

        # Check
        # utils_print.printBrief3Cells(fileNameFull, ["1", "2", "3"], [img, msk, msk])

        # Pack into NumpyArray
        img = np.array([img])
        if saveMask:
            msk = np.array([msk])

        imgs[i] = img
        if saveMask:
            msks[i] = msk

        print('Done: {0}/{1} images'.format(i + 1, total))
        i += 1

    print('Loading done.')

    np.save(conf.pkgDir + label + dataset + '_imgs.npy', imgs)
    if saveMask:
        np.save(conf.pkgDir + label + dataset + '_msks.npy', msks)
    print('Saving to .npy files done.')

def saveLabelledSet(label, dataset, saveMask=True, normalize=True):

    path = os.path.join(conf.setDir, label+'/')

    # Filter out the masks
    images_list = [f for f in os.listdir(path) if re.match('^((?!mask).)*$', f)]
    total = len(images_list)

    # Initialize the numpy arrays
    imgs = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)
    if saveMask:
        msks = np.ndarray((total, conf.img_rows, conf.img_cols, conf.channels), dtype=np.float)

    # Initialize the number of channels flag
    asGray = True
    #if conf.channels == 3:
    #    asGray = False

    print('-' * 30)
    print('Creating training images...')
    print('-' * 30)
    i = 0
    for fileNameFull in images_list:
        fileName = fileNameFull.split(".")[0]
        fileExtension = fileNameFull.split(".")[1]

        if saveMask:
            associated_masks = fnmatch.filter(os.listdir(path), fileName+"_mask*.png")
            print(f"associated: {associated_masks}")

        # Read image and all associated masks
        img = imread(os.path.join(path, fileNameFull), as_gray=asGray)
        if saveMask:
            msk = combineMasks(path, associated_masks, asGray)

        # to Float
        img = img_as_float(img)
        if saveMask:
            if np.max(msk)>0:
                msk = img_as_float(msk / np.max(msk))

        # Resize
        img = resize(img, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)
        if saveMask:
            msk = resize(msk, (conf.img_rows, conf.img_cols, conf.channels), preserve_range=True)

        # Thresholding
        if saveMask:
            msk[msk > 0.5] = 1.0
            msk[msk <= 0.5] = 0.0

        # Check
        #utils_print.printBrief3Cells(fileNameFull, ["1", "2", "3"], [img, msk, msk])

        # Pack into NumpyArray
        img = np.array([img])
        if saveMask:
            msk = np.array([msk])

        # Normalization
        if normalize:
            img -= np.min(img)
            img /= np.max(img)

        # Check Min-Max Values
        #print(f"min:{np.min(img)} - max{np.max(img)}")

        imgs[i] = img
        if saveMask:
            msks[i] = msk

        print(f'Done: {i}/{total} images : {fileName}')
        i += 1
    
    print('Loading done.')

    np.save(conf.pkgDir + label+dataset+'_imgs.npy', imgs)
    if saveMask:
        np.save(conf.pkgDir + label+dataset+'_msks.npy', msks)
    print('Saving to .npy files done.')

def loadLabelledSet(set, loadMask=True):
    if loadMask:
        imgs_set = np.load(conf.pkgDir + set + '_imgs.npy')
        msks_set = np.load(conf.pkgDir + set +'_msks.npy')
        return imgs_set, msks_set
    else:
        imgs_set = np.load(conf.pkgDir + set + '_imgs.npy')
        return imgs_set


def loadPredictions():
    imgs_pred = np.load(conf.pkgDir + 'imgs_prediction.npy')
    return imgs_pred


def getStandardizationParams(set):
    mean = np.mean(set)  # mean for data centering
    std = np.std(set)
    return mean, std

def generateStandardizedSet_Old(set, mean, std):
    set -= mean
    set /= std
    return set

def generateStandardizedSet(set):
    for im in set:
        im -= np.mean(im)
        if np.std(im) != 0:
            im /= np.std(im)

    return set


def generateNormalizedSet(set):
    for im in set:
        im -= np.min(im)
        if np.max(im)>0:
            im /= np.max(im)

    return set


if __name__ == '__main__':
    os.chdir(sys.path[0])
    saveLabelledSet("normal", "BUSI_256", normalize=False)
    saveLabelledSet("benign", "BUSI_256", normalize=False)
    saveLabelledSet("malignant", "BUSI_256", normalize=False)