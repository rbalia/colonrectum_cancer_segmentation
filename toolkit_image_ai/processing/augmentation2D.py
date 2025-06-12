from keras.preprocessing.image import ImageDataGenerator
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


def flip(imgSet, param):
    imgSet_t = imgSet.copy()
    i= 0
    for img in imgSet:
        img = cv2.flip(img, param)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        imgSet_t[i] = img
        i += 1
    return imgSet_t

# Function to distort image
def elastic_transform(image, mask, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """

    # Draw a grid to show the effect
    """# Randomly apply elastic transformation
    def draw_grid(im, grid_size):
        # Draw grid lines
        for i in range(0, im.shape[1], grid_size):
            cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
        for j in range(0, im.shape[0], grid_size):
            cv2.line(im, (0, j), (im.shape[1], j), color=(1,))

    # if np.random.randint(2):
    draw_grid(image, 10)
    draw_grid(mask, 10)"""

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    image = np.expand_dims(image, axis=-1)
    mask = np.expand_dims(mask, axis=-1)

    transformed_img = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    transformed_msk = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)

    transformed_msk[transformed_msk >= 0.5] = 1
    transformed_msk[transformed_msk < 0.5] = 0

    return transformed_img, transformed_msk

def cropInAndResize(img):
    #print(img.shape)
    h, w, ch = img.shape
    img = img[0+15:w-15, 0+15:h-15]
    img = transform.resize(img[0+15:w-15, 0+15:h-15], img.shape ,preserve_range=True)
    return img

def cropOutAndResize(img):
    #print(img.shape)
    img = cv.copyMakeBorder(img, 15, 15, 15, 15, cv.BORDER_CONSTANT, value="black")
    return img

def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.3)
    return img_adapteq

def randomStretch(img):

    """ approach = np.random.randint(2)
    if approach:
        img_logstretch = exposure.equalize_hist(img)
    else: """
    gain = np.random.uniform(0.9, 1.1)
    img_logstretch = exposure.adjust_log(img, gain)

    img_logstretch = np.clip(img_logstretch, 0., 1.)
    return img_logstretch

def dataAugmentationGenerator(img, msk):
    img_aug = np.copy(img)
    msk_aug = np.copy(msk)

    imageGen = ImageDataGenerator(rotation_range=15,
                                  shear_range=0.3,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode="nearest"#"wrap"
                                  )

    params = imageGen.get_random_transform(img.shape)
    img_aug = imageGen.apply_transform(img_aug, params)
    img_aug = randomStretch(img_aug)

    msk_aug = imageGen.apply_transform(msk_aug, params)
    msk_aug[msk_aug >= 0.5] = 1.0
    msk_aug[msk_aug < 0.5] = 0.0

    # Randomly apply elastic transformation
    if np.random.randint(2):
        img_aug, msk_aug = elastic_transform(img_aug, msk_aug,
                                         img_aug.shape[1] * 0.3, img_aug.shape[1] * 0.05, img_aug.shape[1] * 0.05)

    return img_aug, msk_aug

def augmentSetGenerator(img_set, msk_set, n_augmentations, maxCount=0):

    if n_augmentations == 0:
        return img_set, msk_set

    if maxCount > 0:
        total = img_set.shape[0] + maxCount
    else:
        total = img_set.shape[0] * (n_augmentations + 1)

    imgAug_set = np.ndarray((total, img_set.shape[1], img_set.shape[2], img_set.shape[3]), dtype=np.float32)
    mskAug_set = np.ndarray((total, msk_set.shape[1], msk_set.shape[2], msk_set.shape[3]), dtype=np.float32)

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
            imgAug, mskAug = dataAugmentationGenerator(img_set[j], msk_set[j])
            imgAug_set[i] = np.array([imgAug])
            mskAug_set[i] = np.array([mskAug])
            i += 1

            if i >= total:
                break

        j += 1

    return imgAug_set, mskAug_set
