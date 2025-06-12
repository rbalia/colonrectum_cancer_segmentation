import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import numpy as np
import pandas as pd
from src import preprocessing as data
from src import configs as conf
from src import augmentation as uAug
from sklearn.model_selection import KFold
from skimage import io
from matplotlib import pyplot as plt

def getFoldIdx(n_split, set):
    kf = KFold(n_splits=n_split)
    kf.get_n_splits(set)
    foldIdx = []
    for train_index, test_index in kf.split(set):
        foldIdx.append([train_index, test_index])
    return foldIdx

def splitKFold_TestOnly(B_imgs, B_msks, M_imgs, M_msks, N_imgs, N_msks, iteration, seed=None,
               applyStandardization=True, applyNormalization=True, standardizationMode="feature-wise", n_folds=5):

    """
    StandardizationMode define how mean and std are computed: "feature-wise" compute on the whole dataset,
    "sample-wise" compute for each image.
    """

    # Shuffle Data
    if seed is not None:
        B_imgs, B_msks = shuffle(B_imgs, B_msks, random_state=seed)
        M_imgs, M_msks = shuffle(M_imgs, M_msks, random_state=seed)
        N_imgs, N_msks = shuffle(N_imgs, N_msks, random_state=seed)

    imgs = np.concatenate((B_imgs, M_imgs, N_imgs))

    # Split Dataset in Training and Test set
    B_foldIdx = getFoldIdx(n_folds, B_imgs)
    M_foldIdx = getFoldIdx(n_folds, M_imgs)
    N_foldIdx = getFoldIdx(n_folds, N_imgs)

    # Get Training and Test Sets using 5Fold idx
    B_X_test = B_imgs[B_foldIdx[iteration % n_folds][1]]
    B_YS_test  = B_msks[B_foldIdx[iteration % n_folds][1]]

    M_X_test = M_imgs[M_foldIdx[iteration % n_folds][1]]
    M_YS_test = M_msks[M_foldIdx[iteration % n_folds][1]]

    N_X_test = N_imgs[N_foldIdx[iteration % n_folds][1]]
    N_YS_test = N_msks[N_foldIdx[iteration % n_folds][1]]


    splitCount = {
        'Classes': ['benign', 'malignant', 'normal'],
        'Test': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0]]
    }
    df = pd.DataFrame(data=splitCount, index=None)
    print("Number of Test samples:")
    print(df)


    # Combine images sets
    X_test = np.concatenate((B_X_test, M_X_test, N_X_test))

    # Combine masks sets
    YS_test = np.concatenate((B_YS_test, M_YS_test, N_YS_test))

    # Generate Classification Labels
    YC_test = np.concatenate((
        np.repeat("0", B_X_test.shape[0]),
        np.repeat("1", M_X_test.shape[0]),
        np.repeat("2", N_X_test.shape[0])))

    YC_test = tf.keras.utils.to_categorical(YC_test, num_classes=3)


    # NORMALIZATION & STANDARDIZATION ==================================================================================================
    X_test_std = X_test.copy()

    if applyNormalization:
        print('-' * conf.dlw)
        print('Dataset Normalization...')
        print('-' * conf.dlw)

        X_test_std = data.generateNormalizedSet(X_test_std)

    if applyStandardization:
        print('-' * conf.dlw)
        print(f'Dataset Standardization | Mode: {standardizationMode}')
        print('-' * conf.dlw)

        if standardizationMode == "feature-wise":
            mean, std = data.getStandardizationParams(imgs)
            X_test_std = data.generateStandardizedSet_Old(X_test_std, mean, std)

        elif standardizationMode == "sample-wise":
            X_test_std = data.generateStandardizedSet(X_test_std)

        else:
            print("Illegal standardization mode: choose between ['feature-wise', 'sample-wise']")
            exit(0)


    return X_test, YS_test, YC_test, X_test_std



def splitKFold(B_imgs, B_msks, M_imgs, M_msks, N_imgs, N_msks, iteration, seed=None,
               applyStandardization=True, applyNormalization=True, standardizationMode="feature-wise", n_folds=5):

    """
    StandardizationMode define how mean and std are computed:
        -   "feature-wise" compute on the whole dataset,
        -   "sample-wise" compute for each image.
    """

    # Shuffle Data
    if seed is not None:
        B_imgs, B_msks = shuffle(B_imgs, B_msks, random_state=seed)
        M_imgs, M_msks = shuffle(M_imgs, M_msks, random_state=seed)
        N_imgs, N_msks = shuffle(N_imgs, N_msks, random_state=seed)

    imgs = np.concatenate((B_imgs, M_imgs, N_imgs))

    # Split Dataset in Training and Test set
    B_foldIdx = getFoldIdx(n_folds, B_imgs)
    M_foldIdx = getFoldIdx(n_folds, M_imgs)
    N_foldIdx = getFoldIdx(n_folds, N_imgs)

    # Get Training and Test Sets using 5Fold idx
    B_X_train_val = B_imgs[B_foldIdx[iteration % n_folds][0]]
    B_YS_train_val  = B_msks[B_foldIdx[iteration % n_folds][0]]
    B_X_test = B_imgs[B_foldIdx[iteration % n_folds][1]]
    B_YS_test  = B_msks[B_foldIdx[iteration % n_folds][1]]

    M_X_train_val = M_imgs[M_foldIdx[iteration % n_folds][0]]
    M_YS_train_val = M_msks[M_foldIdx[iteration % n_folds][0]]
    M_X_test = M_imgs[M_foldIdx[iteration % n_folds][1]]
    M_YS_test = M_msks[M_foldIdx[iteration % n_folds][1]]

    N_X_train_val = N_imgs[N_foldIdx[iteration % n_folds][0]]
    N_YS_train_val = N_msks[N_foldIdx[iteration % n_folds][0]]
    N_X_test = N_imgs[N_foldIdx[iteration % n_folds][1]]
    N_YS_test = N_msks[N_foldIdx[iteration % n_folds][1]]

    # Split Dataset in Training and Validation set
    B_X_train, B_X_val, B_YS_train, B_YS_val = train_test_split(B_X_train_val, B_YS_train_val, test_size=0.2,
                                                                shuffle=False, random_state=seed)
    M_X_train, M_X_val, M_YS_train, M_YS_val = train_test_split(M_X_train_val, M_YS_train_val, test_size=0.2,
                                                                shuffle=False, random_state=seed)
    N_X_train, N_X_val, N_YS_train, N_YS_val = train_test_split(N_X_train_val, N_YS_train_val, test_size=0.2,
                                                                shuffle=False, random_state=seed)


    splitCount = {
        'Dataset': ['benign', 'malignant', 'normal', 'TOTAL'],
        'Training': [B_X_train.shape[0], M_X_train.shape[0], N_X_train.shape[0], 0],
        'Validation': [B_X_val.shape[0], M_X_val.shape[0], N_X_val.shape[0], 0],
        'Test': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0], 0]
    }
    df = pd.DataFrame(data=splitCount, index=None)
    print("Number of samples BEFORE augmentation:")
    print(df)

    """# GENERATE AUGMENTATION
    B_X_train,B_YS_train = uAug.augmentSetGenerator(B_X_train,B_YS_train, 3, maxCount=0)#+1
    M_X_train, M_YS_train = uAug.augmentSetGenerator(M_X_train,M_YS_train, 7, maxCount=0)#+2
    N_X_train, N_YS_train = uAug.augmentSetGenerator(N_X_train,N_YS_train, 10, maxCount=0)#+3

    B_X_val, B_YS_val = uAug.augmentSetGenerator(B_X_val, B_YS_val, 0, maxCount=0) #0
    M_X_val, M_YS_val = uAug.augmentSetGenerator(M_X_val, M_YS_val, 1, maxCount=0) #1
    N_X_val, N_YS_val = uAug.augmentSetGenerator(N_X_val, N_YS_val, 2, maxCount=0) #2"""

    """B_X_train,B_YS_train = uAug.augmentSetGenerator(B_X_train,B_YS_train, 8, maxCount=0)#+1
    M_X_train, M_YS_train = uAug.augmentSetGenerator(M_X_train,M_YS_train, 8, maxCount=0)#+2
    N_X_train, N_YS_train = uAug.augmentSetGenerator(N_X_train,N_YS_train, 8, maxCount=0)#+3

    B_X_val, B_YS_val = uAug.augmentSetGenerator(B_X_val, B_YS_val, 5, maxCount=0)
    M_X_val, M_YS_val = uAug.augmentSetGenerator(M_X_val, M_YS_val, 5, maxCount=0)
    N_X_val, N_YS_val = uAug.augmentSetGenerator(N_X_val, N_YS_val, 5, maxCount=0)"""


    splitCount = {
        'Dataset': ['benign', 'malignant', 'normal', 'TOTAL'],
        'Training': [B_X_train.shape[0], M_X_train.shape[0], N_X_train.shape[0], 0],
        'Validation': [B_X_val.shape[0], M_X_val.shape[0], N_X_val.shape[0], 0],
        'Test': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0], 0]
    }
    df = pd.DataFrame(data=splitCount, index=None)
    print("Number of samples AFTER augmentation:")
    print(df)


    # Combine images sets
    X_train = np.concatenate((B_X_train, M_X_train, N_X_train))
    X_validation = np.concatenate((B_X_val, M_X_val, N_X_val))
    X_test = np.concatenate((B_X_test, M_X_test, N_X_test))

    # Combine masks sets
    YS_train = np.concatenate((B_YS_train, M_YS_train, N_YS_train))
    YS_validation = np.concatenate((B_YS_val, M_YS_val, N_YS_val))
    YS_test = np.concatenate((B_YS_test, M_YS_test, N_YS_test))

    # Generate Classification Labels
    YC_train = np.concatenate((
        np.repeat("0", B_X_train.shape[0]),
        np.repeat("1", M_X_train.shape[0]),
        np.repeat("2", N_X_train.shape[0])))
    YC_validation = np.concatenate((
        np.repeat("0", B_X_val.shape[0]),
        np.repeat("1", M_X_val.shape[0]),
        np.repeat("2", N_X_val.shape[0])))
    YC_test = np.concatenate((
        np.repeat("0", B_X_test.shape[0]),
        np.repeat("1", M_X_test.shape[0]),
        np.repeat("2", N_X_test.shape[0])))

    # Shuffle
    X_train, YS_train, YC_train = shuffle(X_train, YS_train, YC_train, random_state=seed)
    X_validation, YS_validation, YC_validation = shuffle(X_validation, YS_validation, YC_validation, random_state=seed)

    # Convert to categorical labels array
    YC_train = tf.keras.utils.to_categorical(YC_train, num_classes=3)
    YC_validation = tf.keras.utils.to_categorical(YC_validation, num_classes=3)
    YC_test = tf.keras.utils.to_categorical(YC_test, num_classes=3)

    # Y_validation = tf.constant(Y_validation, shape=(Y_validation.shape[0], 3))
    # Y_train = tf.constant(Y_train, shape=(Y_train.shape[0],3))
    # YC_test = tf.constant(YC_test, shape=(YC_test.shape[0], 3))

    # NORMALIZATION & STANDARDIZATION ==================================================================================================
    X_test_std = X_test.copy()

    if applyNormalization:
        print('-' * conf.dlw)
        print('Dataset Normalization...')
        print('-' * conf.dlw)

        X_train = data.generateNormalizedSet(X_train)
        X_validation = data.generateNormalizedSet(X_validation)
        X_test_std = data.generateNormalizedSet(X_test_std)

    if applyStandardization:
        print('-' * conf.dlw)
        print(f'Dataset Standardization | Mode: {standardizationMode}')
        print('-' * conf.dlw)

        if standardizationMode == "feature-wise":
            mean, std = data.getStandardizationParams(imgs)
            X_train = data.generateStandardizedSet_Old(X_train, mean, std)
            X_validation = data.generateStandardizedSet_Old(X_validation, mean, std)
            X_test_std = data.generateStandardizedSet_Old(X_test_std, mean, std)

        elif standardizationMode == "sample-wise":
            X_train = data.generateStandardizedSet(X_train)
            X_validation = data.generateStandardizedSet(X_validation)
            X_test_std = data.generateStandardizedSet(X_test_std)

        else:
            print("Illegal standardization mode: choose between ['feature-wise', 'sample-wise']")
            exit(0)


    return X_train, YS_train, YC_train, \
           X_validation, YS_validation, YC_validation, \
           X_test, YS_test, YC_test, X_test_std


def splitKFold_OLD(B_imgs, B_msks, M_imgs, M_msks, N_imgs, N_msks, seed, iteration,
               applyStandardization=True, applyNormalization=True, n_folds=5,
               deprecatedApproach=False, deleteAnomalies=False):

    # Shuffle Data
    if seed is not None:
        B_imgs, B_msks = shuffle(B_imgs, B_msks, random_state=seed)
        M_imgs, M_msks = shuffle(M_imgs, M_msks, random_state=seed)
        N_imgs, N_msks = shuffle(N_imgs, N_msks, random_state=seed)

    imgs = np.concatenate((B_imgs, M_imgs, N_imgs))

    # SPLIT DATASET
    B_foldIdx = getFoldIdx(n_folds, B_imgs)
    M_foldIdx = getFoldIdx(n_folds, M_imgs)
    N_foldIdx = getFoldIdx(n_folds, N_imgs)

    # Remove Anomalies
    test_anomalies = []

    if deleteAnomalies:
        B_anomalies = [333, 167, 356, 395, 421, 431, 10, 17, 19, 22, 26, 36, 60, 72, 110, 114, 149,
                       152, 159, 164, 165, 188, 209, 221, 226, 247, 260, 327, 332, 337, 338, 341,
                       371, 373, 375]
        M_anomalies = [89, 157, 158, 185, 186, 203, 204, 8, 9, 19, 51]
        N_anomalies = [0, 61]

        B_train_arr = B_foldIdx[iteration % n_folds][0]
        B_test_arr = B_foldIdx[iteration % n_folds][1]
        B_test_len = len(B_test_arr)

        M_train_arr = M_foldIdx[iteration % n_folds][0]
        M_test_arr = M_foldIdx[iteration % n_folds][1]
        M_test_len = len(M_test_arr)

        N_train_arr = N_foldIdx[iteration % n_folds][0]
        N_test_arr = N_foldIdx[iteration % n_folds][1]
        N_test_len = len(N_test_arr)

        for x in B_anomalies:

            if len(np.argwhere(B_foldIdx[iteration % n_folds][1] == x)) > 0:
                id_anomaly_test = np.argwhere(B_foldIdx[iteration % n_folds][1] == x).flat[0]
                test_anomalies.append(id_anomaly_test)
            B_train_arr = np.delete(B_train_arr, np.argwhere(B_train_arr == x))
            B_test_arr = np.delete(B_test_arr, np.argwhere(B_test_arr == x))
        for x in M_anomalies:
            if len(np.argwhere(M_foldIdx[iteration % n_folds][1] == x)) > 0:
                id_anomaly_test = np.argwhere(M_foldIdx[iteration % n_folds][1] == x).flat[0]
                print(f"pre: {id_anomaly_test}")
                id_anomaly_test += B_test_len
                id_anomaly_test += 0
                print(f"post: {id_anomaly_test}")
                test_anomalies.append(id_anomaly_test)
            M_train_arr = np.delete(M_train_arr, np.argwhere(M_train_arr == x))
            M_test_arr = np.delete(M_test_arr, np.argwhere(M_test_arr == x))
        for x in N_anomalies:
            if len(np.argwhere(N_foldIdx[iteration % n_folds][1] == x)) > 0:
                id_anomaly_test = np.argwhere(N_foldIdx[iteration % n_folds][1] == x).flat[0]
                id_anomaly_test += B_test_len + M_test_len
                id_anomaly_test += 0
                test_anomalies.append(id_anomaly_test)
            N_train_arr = np.delete(N_train_arr, np.argwhere(N_train_arr == x))
            N_test_arr = np.delete(N_test_arr, np.argwhere(N_test_arr == x))

        print(test_anomalies)

        B_foldIdx[iteration % n_folds][0] = B_train_arr
        B_foldIdx[iteration % n_folds][1] = B_test_arr
        M_foldIdx[iteration % n_folds][0] = M_train_arr
        M_foldIdx[iteration % n_folds][1] = M_test_arr
        N_foldIdx[iteration % n_folds][0] = N_train_arr
        N_foldIdx[iteration % n_folds][1] = N_test_arr

    # Get Images using 5Fold idx
    B_X_train = B_imgs[B_foldIdx[iteration % n_folds][0]]
    B_YS_train  = B_msks[B_foldIdx[iteration % n_folds][0]]
    B_X_test = B_imgs[B_foldIdx[iteration % n_folds][1]]
    B_YS_test  = B_msks[B_foldIdx[iteration % n_folds][1]]

    M_X_train = M_imgs[M_foldIdx[iteration % n_folds][0]]
    M_YS_train = M_msks[M_foldIdx[iteration % n_folds][0]]
    M_X_test = M_imgs[M_foldIdx[iteration % n_folds][1]]
    M_YS_test = M_msks[M_foldIdx[iteration % n_folds][1]]

    N_X_train = N_imgs[N_foldIdx[iteration % n_folds][0]]
    N_YS_train = N_msks[N_foldIdx[iteration % n_folds][0]]
    N_X_test = N_imgs[N_foldIdx[iteration % n_folds][1]]
    N_YS_test = N_msks[N_foldIdx[iteration % n_folds][1]]

    splitCount = {
        'Dataset': ['benign', 'malignant', 'normal', 'TOTAL'],
        'Training': [B_X_train.shape[0], M_X_train.shape[0], N_X_train.shape[0], 0],
        'Validation': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0], 0],
        'Test': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0], 0]
    }
    df = pd.DataFrame(data=splitCount, index=None)
    print("Number of samples BEFORE augmentation:")
    print(df)

    # GENERATE AUGMENTATION
    B_X_train,B_YS_train = uAug.augmentSetGenerator(B_X_train,B_YS_train, 3, maxCount=0)#+1
    M_X_train, M_YS_train = uAug.augmentSetGenerator(M_X_train,M_YS_train, 7, maxCount=0)#+2
    N_X_train, N_YS_train = uAug.augmentSetGenerator(N_X_train,N_YS_train, 8, maxCount=0)#+3

    B_X_validation, B_YS_validation = uAug.augmentSetGenerator(B_X_test, B_YS_test, 0, maxCount=0)
    M_X_validation, M_YS_validation = uAug.augmentSetGenerator(M_X_test, M_YS_test, 1, maxCount=0)
    N_X_validation, N_YS_validation = uAug.augmentSetGenerator(N_X_test, N_YS_test, 2, maxCount=0)

    splitCount = {
        'Dataset': ['benign', 'malignant', 'normal', 'TOTAL'],
        'Training': [B_X_train.shape[0], M_X_train.shape[0], N_X_train.shape[0], 0],
        'Validation': [B_X_validation.shape[0], M_X_validation.shape[0], N_X_validation.shape[0], 0],
        'Test': [B_X_test.shape[0], M_X_test.shape[0], N_X_test.shape[0], 0]
    }
    df = pd.DataFrame(data=splitCount, index=None)
    print("Number of samples AFTER augmentation:")
    print(df)


    # Combine images sets
    X_train = np.concatenate((B_X_train, M_X_train, N_X_train))
    X_validation = np.concatenate((B_X_validation, M_X_validation, N_X_validation))
    X_test = np.concatenate((B_X_test, M_X_test, N_X_test))

    # Combine masks sets
    YS_train = np.concatenate((B_YS_train, M_YS_train, N_YS_train))
    YS_validation = np.concatenate((B_YS_validation, M_YS_validation, N_YS_validation))
    YS_test = np.concatenate((B_YS_test, M_YS_test, N_YS_test))

    # Generate Classification Labels
    YC_train = np.concatenate((
        np.repeat("0", B_X_train.shape[0]),
        np.repeat("1", M_X_train.shape[0]),
        np.repeat("2", N_X_train.shape[0])))
    YC_validation = np.concatenate((
        np.repeat("0", B_X_validation.shape[0]),
        np.repeat("1", M_X_validation.shape[0]),
        np.repeat("2", N_X_validation.shape[0])))
    YC_test = np.concatenate((
        np.repeat("0", B_X_test.shape[0]),
        np.repeat("1", M_X_test.shape[0]),
        np.repeat("2", N_X_test.shape[0])))

    # Shuffle
    X_train, YS_train, YC_train = shuffle(X_train, YS_train, YC_train, random_state=None)
    X_validation, YS_validation, YC_validation = shuffle(X_validation, YS_validation, YC_validation, random_state=None)

    # Convert to categorical labels array
    YC_train = tf.keras.utils.to_categorical(YC_train, num_classes=3)
    YC_validation = tf.keras.utils.to_categorical(YC_validation, num_classes=3)
    YC_test = tf.keras.utils.to_categorical(YC_test, num_classes=3)

    # Y_validation = tf.constant(Y_validation, shape=(Y_validation.shape[0], 3))
    # Y_train = tf.constant(Y_train, shape=(Y_train.shape[0],3))
    YC_test = tf.constant(YC_test, shape=(YC_test.shape[0], 3))

    # NORMALIZATION & STANDARDIZATION ==================================================================================================
    X_test_std = X_test.copy()
    if deprecatedApproach:
        mean, std = data.getStandardizationParams(imgs)
        X_train = data.generateStandardizedSet_Old(X_train,mean, std)
        X_validation = data.generateStandardizedSet_Old(X_validation,mean, std)
        X_test_std = data.generateStandardizedSet_Old(X_test_std,mean, std)
    else:
        if applyNormalization:
            print('-' * conf.dlw)
            print('Dataset Normalization...')
            print('-' * conf.dlw)

            X_train = data.generateNormalizedSet(X_train)
            X_validation = data.generateNormalizedSet(X_validation)
            X_test_std = data.generateNormalizedSet(X_test_std)

        if applyStandardization:
            print('-' * conf.dlw)
            print('Dataset Standardization...')
            print('-' * conf.dlw)

            X_train = data.generateStandardizedSet(X_train)
            X_validation = data.generateStandardizedSet(X_validation)
            X_test_std = data.generateStandardizedSet(X_test_std)


    return X_train, YS_train, YC_train, \
           X_validation, YS_validation, YC_validation, \
           X_test, YS_test, YC_test, X_test_std, test_anomalies
