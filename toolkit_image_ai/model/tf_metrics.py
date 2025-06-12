from __future__ import print_function
import tensorflow as tf
import tensorflow.keras.backend as K
smooth = 1e-7

def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou_coef_thresholded(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)#, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred) - intersection
    iou = tf.reduce_mean(intersection / (union + smooth_den))
    return iou

def iou_coef_thresholded_tf(y_true, y_pred, tr=0.5):
    def iou_metric_np(y_true, y_pred):
        return iou_coef_thresholded(y_true, y_pred, tr).numpy()
    return tf.py_function(iou_metric_np, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_thresholded(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    dice = 2 * intersection / (sum_ + smooth_den)

    return dice

def dice_coef_thresholded_tf(y_true, y_pred, tr=0.5):
    def dice_metric_np(y_true, y_pred):
        return dice_coef_thresholded(y_true, y_pred, tr).numpy()
    return tf.py_function(dice_metric_np, [y_true, y_pred], tf.float32)

def tpr_thresholded(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true)
    tpr = intersection / (sum_ + smooth_den)

    return tpr

def ppv_thresholded(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    fp = tf.reduce_sum((1.0-y_true) * y_pred)
    sum_ = tf.reduce_sum(y_pred)
    fdr = fp / (sum_ + smooth_den)

    return 1.0 - fdr

def gMean(y_true, y_pred):
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f) + smooth
    ppv = true_positives / (tf.reduce_sum(y_pred_f) + smooth)
    tpr = true_positives / (tf.reduce_sum(y_true_f) + smooth)
    return tf.math.sqrt(tpr * ppv)

