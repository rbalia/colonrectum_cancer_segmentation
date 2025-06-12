import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import warnings
from toolkit_image_ai.model.prediction import metrics as local_metrics


def evaluate_segmentation(y_true, y_pred, t=0.5, det_t=0.5):
    iou_scores = []
    dice_scores = []
    intersection_scores = []
    tnr_score = []
    falsediscovery_score = []
    mcc_score = []
    fmi_score = []
    detection_pred = []
    lesion_detection_pred = []
    normal_detection_pred = []
    detection_truth = []

    for truth, pred in zip(y_true, y_pred):
        pred[pred >= t] = 1
        pred[pred < t] = 0

        curr_IOU = local_metrics.evaluateIOU(pred, truth)
        curr_DICE = local_metrics.evaluateDice(pred, truth)
        curr_intersection = local_metrics.evaluateTruePositiveRate(pred, truth)
        curr_tnr = local_metrics.evaluateTrueNegativeRate(pred, truth)
        curr_falsepositive = local_metrics.evaluateFalseDiscoveryRate(pred,truth)
        curr_mcc = 0#metrics.evaluateMatthewsCorrelationCoefficient(pred, truth)#evaluateF1Score(pred,truth)#0#
        curr_fmi = local_metrics.evaluateFowlkesMallowsIndex(pred,truth)

        iou_scores.append(curr_IOU)
        dice_scores.append(curr_DICE)
        intersection_scores.append(curr_intersection)
        tnr_score.append(curr_tnr)
        falsediscovery_score.append(curr_falsepositive)
        mcc_score.append(curr_mcc)
        fmi_score.append(curr_fmi)

        if np.count_nonzero(pred) == 0:
            detection_pred.append(0)
        else:
            if np.count_nonzero(pred) > 0 and curr_IOU >= det_t:
                detection_pred.append(1)
            else:
                detection_pred.append(-1)

        if np.count_nonzero(truth) == 0:
            detection_truth.append(0)
            # Normal class detection
            if np.count_nonzero(pred) == 0:
                normal_detection_pred.append(1)
            else:
                normal_detection_pred.append(0)
        else:
            detection_truth.append(1)
            # Lesion class detection
            if np.count_nonzero(pred) > 0 and curr_IOU >= det_t:
                lesion_detection_pred.append(1)
            else:
                lesion_detection_pred.append(0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        detection_accuracy = metrics.accuracy_score(detection_truth, detection_pred)
        detection_bal_acc = metrics.balanced_accuracy_score(detection_truth, detection_pred)

    #tn, fp, fn, tp = metrics.confusion_matrix(detection_truth,detection_pred).ravel()
    #tnr = tn / (tn + fp) # specificity
    #tpr = tp / (tp + fn) # sensitivity
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tnr = np.sum(normal_detection_pred) / len(normal_detection_pred)
        tpr = np.sum(lesion_detection_pred) / len(lesion_detection_pred)


    return {"iou":iou_scores, "dice":dice_scores, "tnr": tnr_score, "tpr":intersection_scores,
            "fdr":falsediscovery_score, "det-acc":detection_accuracy, "det-bacc":detection_bal_acc,
            "det-tnr":tnr, "det-tpr":tpr, "mcc":mcc_score, "fmi": fmi_score}

def evaluate_classification(y_true, y_pred, n_classes, verbose=False, modelName="unknown"):

    count_p = [0] * n_classes
    count_t = [0] * n_classes
    correctPred = 0

    decoded_y_true = []
    decoded_y_pred = []

    errorNaN = False
    if y_pred is not None:
        for pred, truth in zip(y_pred, y_true):
            if np.isnan(max(pred)):
                max_index_pred = 0
                errorNaN = True
            else:
                max_index_pred = pred.tolist().index(max(pred))

            max_index_truth = truth.tolist().index(max(truth))

            decoded_y_true.append(max_index_truth)
            decoded_y_pred.append(max_index_pred)

            if (max_index_pred == max_index_truth):
                correctPred += 1

            count_p[max_index_pred] += 1
            count_t[max_index_truth] += 1


    if errorNaN: print("ERROR: NaN values detected on predictions probability")

    matrix = confusion_matrix(y_pred.argmax(axis=1), y_true.argmax(axis=1))
    matrix = np.array(matrix).transpose()
    #df_val = {'B': matrix[0], 'M': matrix[1], 'N': matrix[2]}
    dataFrame_confusionMatrix = pd.DataFrame(data=matrix)

    accuracy = metrics.accuracy_score(decoded_y_true, decoded_y_pred)#round((correctPred / y_true.shape[0]), 4)
    bal_acc = metrics.balanced_accuracy_score(decoded_y_true, decoded_y_pred)  # round((correctPred / y_true.shape[0]), 4)
    precision = metrics.precision_score(decoded_y_true, decoded_y_pred, average="weighted")
    recall = metrics.recall_score(decoded_y_true, decoded_y_pred, average="weighted")
    f1score = metrics.f1_score(decoded_y_true, decoded_y_pred, average="weighted")

    if verbose:
        print(f"{modelName} | Count Predictions : " + str(count_p))
        print(f"{modelName} | Count Real Values : " + str(count_t))
        print(f"{modelName} | Accuracy: " + str(accuracy) + "%")
        print(f"{modelName} | Balanced Accuracy: " + str(bal_acc) + "%")
        print(f"{modelName} | Confusion Matrix: ")
        print(dataFrame_confusionMatrix)
        #confusionMatrix.pretty_plot_confusion_matrix(dataFrame_confusionMatrix, pred_val_axis="y", modelName="ResNet50")
        print('-' * 50)

    return {"acc": accuracy, "bacc":bal_acc, "pre": precision, "rec": recall, "f1s": f1score,
            "decoded_pred": decoded_y_pred, "decoded_true": decoded_y_true,
            "dfConfMatrix": dataFrame_confusionMatrix, "countPreds": count_p, "countTruth": count_t}