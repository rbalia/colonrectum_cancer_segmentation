from math import sqrt
import numpy as np
import tensorflow as tf

def evaluateIOU(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    iou_score = (np.sum(intersection) + 1.) / (np.sum(union) + 1.)
    return round(iou_score, 2)

def evaluateDice(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.sum(prediction) + np.sum(label)
    dice_score = ((2 * np.sum(intersection)) + 1.) / (union + 1.)
    return round((dice_score), 2)

def evaluateTruePositiveRate(prediction, label):
    intersection = np.logical_and(prediction, label)
    intersection_score = (np.sum(intersection)) / (np.sum(label) + 0.0001)
    return intersection_score

def evaluateTrueNegativeRate(prediction, label):
    TN = np.logical_and(1 - prediction, 1 - label)
    FP = np.logical_and(prediction, 1 - label)
    tnr_score = (np.sum(TN)) / ((np.sum(TN)) + (np.sum(FP)))
    return tnr_score

def evaluateFalseDiscoveryRate(prediction, label):
    intersection_negative = np.logical_and(prediction, 1-label)
    fp_score = (np.sum(intersection_negative)) / (np.sum(prediction)+ 0.0001)
    return fp_score

def evaluateF1Score(prediction, label):
    ppv = 1-evaluateFalseDiscoveryRate(prediction, label)
    tpr = evaluateTruePositiveRate(prediction, label)
    if tpr + ppv == 0:
        return 0
    else:
        f1s = 2*((tpr * ppv) / (tpr + ppv))
        return round(f1s, 2)

def evaluateFowlkesMallowsIndex(prediction,label):
    # Metodo 1
    ppv = 1 - evaluateFalseDiscoveryRate(prediction, label)
    tpr = evaluateTruePositiveRate(prediction, label)
    fmi = sqrt(tpr * ppv)
    return fmi

    # Metodo 2 (equivalente a 1)
    #tp = np.sum(label * prediction)
    #fp = np.sum((1 - label) * prediction)
    #fn = np.sum(label * (1 - prediction))
    #fmi = np.sqrt((tp / (tp + fp)) * (tp / (tp + fn)))
    #return fmi

def evaluateMatthewsCorrelationCoefficient(y_pred, y_true, epsilon=1e-5):
    """
    Computes the Matthews Correlation Coefficient loss between y_true and y_pred.
    """
    # TODO: Replace tf with np
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_pred_f = tf.keras.backend.round(y_pred_f)

    true_positives = tf.keras.backend.sum(y_true_f * y_pred_f)
    true_negatives = tf.keras.backend.sum((1 - y_true_f) * (1 - y_pred_f))
    false_positives = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    false_negatives = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))

    numerator = true_positives * true_negatives - false_positives * false_negatives
    denominator = tf.keras.backend.sqrt(
        (true_positives + false_positives) * (true_positives + false_negatives) * (
                true_negatives + false_positives) * (true_negatives + false_negatives))
    mcc = numerator / (denominator + epsilon)

    #ppv = 1-(false_positives/(tf.keras.backend.sum(y_pred_f)))
    #tpr = true_positives/(tf.keras.backend.sum(y_true_f))
    return mcc#tf.math.sqrt(tpr*ppv)

def customModelScore(iou, dice, tpr, ppv):
    return ((((iou + dice) / 2) - (dice - iou)) + tpr + ppv) / 3
def genericGeometricMean(tpr, ppv):
    return sqrt(tpr*ppv)