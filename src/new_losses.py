import tensorflow as tf

def iou_metric(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)#, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred) - intersection
    iou = tf.reduce_mean(intersection / (union + smooth_den))

    return iou

def iou_metric_tf(y_true, y_pred, tr=0.5):
    def iou_metric_np(y_true, y_pred):
        return iou_metric(y_true, y_pred, tr).numpy()
    return tf.py_function(iou_metric_np, [y_true, y_pred], tf.float32)

def dice_metric(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true + y_pred)
    dice = 2 * intersection / (sum_ + smooth_den)

    return dice

def metric_tpr(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    sum_ = tf.reduce_sum(y_true)
    tpr = intersection / (sum_ + smooth_den)

    return tpr

def metric_ppv(y_true, y_pred, tr=0.5):
    smooth_den = 1e-7
    y_true = tf.cast(y_true > tr, tf.float32)
    y_pred = tf.cast(y_pred > tr, tf.float32)

    fp = tf.reduce_sum((1.0-y_true) * y_pred)
    sum_ = tf.reduce_sum(y_pred)
    fdr = fp / (sum_ + smooth_den)

    return 1.0 - fdr

def dice_metric_tf(y_true, y_pred, tr=0.5):
    def dice_metric_np(y_true, y_pred):
        return dice_metric(y_true, y_pred, tr).numpy()
    return tf.py_function(dice_metric_np, [y_true, y_pred], tf.float32)

def custom_loss_TPR_FDR(y_true, y_pred):
    smooth_den = 1e-7

    intersection = tf.reduce_sum(y_true * y_pred)
    intersection_score = intersection / (tf.reduce_sum(y_true) + smooth_den)

    intersection_negative = tf.reduce_sum(y_true * (1-y_pred))
    fp_score = (intersection_negative) / (tf.reduce_sum(y_pred) + smooth_den)

    return 1.-((intersection_score + (1-fp_score))/2)