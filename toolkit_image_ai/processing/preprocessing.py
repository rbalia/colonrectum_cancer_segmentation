import numpy as np
#import skimage

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
    #nparray = skimage.color.gray2rgb(nparray) #funzionante testato
    nparray = np.repeat(nparray, 3, axis=-1)
    return nparray

def setColor2Gray(nparray):
    nparray = nparray[:, :, :, 0]
    nparray = np.expand_dims(nparray, axis=-1)
    return nparray