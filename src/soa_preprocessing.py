import numpy as np
import plotly.graph_objects as go
import torch
import plotly.subplots as sp
from math import sqrt
import pandas as pd
from skimage import io
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def compute_subcubes_count(cube_size, subcube_size, stride):
    num_subcubes = ((cube_size[0] - subcube_size[0]) // stride + 1) * \
                   ((cube_size[1] - subcube_size[1]) // stride + 1) * \
                   ((cube_size[2] - subcube_size[2]) // stride + 1)
    #print(f"Numero sottocubi estraibili: {num_subcubes}")
    return num_subcubes

def extract_subcubes(X, subcube_size, stride):

    # Calcoliamo il numero totale di sottocubi che possiamo estrarre dall'immagine
    num_subcubes = compute_subcubes_count(X.shape, subcube_size, stride)

    # Inizializziamo un array per contenere i sottocubi estratti
    subcubes = np.zeros((num_subcubes,) + subcube_size)

    # Estraiamo i sottocubi utilizzando la finestra scorrevole
    subcube_idx = 0
    for i in range(0, X.shape[0] - subcube_size[0] + 1, stride):
        for j in range(0, X.shape[1] - subcube_size[1] + 1, stride):
            for k in range(0, X.shape[2] - subcube_size[2] + 1, stride):
                subcube = X[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :]
                subcubes[subcube_idx, :, :, :, :] = subcube
                subcube_idx += 1
    return subcubes


def reconstruct_image(subcubes, image_shape, subcube_size, stride, mode="avg"):
    """
    Ricostruisce l'immagine originale a partire dai sottocubi estratti.

    Args:
        subcubes (ndarray): array contenente i sottocubi estratti dall'immagine originale
        image_shape (tuple): tupla contenente le dimensioni dell'immagine originale
        subcube_size (tuple): tupla contenente le dimensioni dei sottocubi
        stride (int): passo di spostamento della finestra scorrevole utilizzata per estrarre i sottocubi

    Returns:
        ndarray: array contenente l'immagine ricostruita
    """
    subcubes = subcubes.transpose(0, 2, 3, 4, 1)
    # Inizializza l'immagine ricostruita con zeri
    image = np.zeros(image_shape)

    # Inizializza un array che conta il numero di sovrapposizioni di ogni voxel
    overlap_count = np.zeros(image_shape)
    subcube_idx = 0
    # Scorri lungo i sottocubi e aggiungi ogni sottocubo all'immagine ricostruita
    for i in range(0, image_shape[0] - subcube_size[0] + 1, stride):
        for j in range(0, image_shape[1] - subcube_size[1] + 1, stride):
            for k in range(0, image_shape[2] - subcube_size[2] + 1, stride):
                subcube = subcubes[subcube_idx, :, :, :, :]
                if mode=="avg":
                
                    image[i:i+subcube_size[0], j:j+subcube_size[1], k:k+subcube_size[2], :] += subcube
                    overlap_count[i:i+subcube_size[0], j:j+subcube_size[1], k:k+subcube_size[2], :] += 1
                elif mode=="max":
                    image[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :] = \
                        np.maximum(image[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :],subcube)
                subcube_idx += 1

    # Divide ogni voxel dell'immagine ricostruita per il numero di sovrapposizioni per ottenere la media
    if mode == "avg":
        image /= overlap_count

    return image

def plot_3d_interactive(pred_mask, mask, dice_score, threshold=0.5):
    """
    Visualizza pred_mask e mask in due grafici interattivi 3D affiancati con colori in gradiente.
    
    Args:
        pred_mask (torch.Tensor or np.ndarray): Maschera predetta dal modello.
        mask (torch.Tensor or np.ndarray): Maschera ground truth.
        dice_score (float): Valore del Dice Score da mostrare.
        threshold (float): Valore di soglia per considerare un voxel attivo.
    """
    # Converti in numpy se è un tensore PyTorch
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    pred_mask = np.squeeze(pred_mask)  # Rimuove dimensioni inutili
    mask = np.squeeze(mask)

    # Trova le coordinate dei voxel attivi
    pred_x, pred_y, pred_z = np.where(pred_mask > threshold)
    mask_x, mask_y, mask_z = np.where(mask > threshold)

    # Creazione dei due subplot affiancati
    fig = sp.make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Predicted Mask", "Ground Truth Mask"),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # Aggiungi la maschera predetta con colore in gradiente (Z -> Profondità)
    fig.add_trace(go.Scatter3d(
        x=pred_x, y=pred_y, z=pred_z,
        mode='markers',
        marker=dict(size=2, color=pred_z, colorscale='Viridis', opacity=0.6),
        name='Predicted Mask'
    ), row=1, col=1)

    # Aggiungi la maschera ground truth con colore in gradiente (Z -> Profondità)
    fig.add_trace(go.Scatter3d(
        x=mask_x, y=mask_y, z=mask_z,
        mode='markers',
        marker=dict(size=2, color=mask_z, colorscale='Cividis', opacity=0.6),
        name='Ground Truth Mask'
    ), row=1, col=2)

    # Configura il layout con Dice Score
    fig.update_layout(
        title=f"<b>3D Mask Comparison</b><br><span style='font-size:20px;'>Dice Score: {dice_score:.4f}</span>",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        scene2=dict(  # Configura la seconda scena (Ground Truth)
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.show()

def evaluate_dice_new(prediction, label):
    pred_mask = (prediction > 0.5).astype(np.uint8)  # Binarizza
    mask = (label > 0.5).astype(np.uint8)            # Binarizza
    intersection = np.logical_and(pred_mask, mask)
    union = np.sum(pred_mask) + np.sum(mask)
    dice_score = ((2 * np.sum(intersection)) + .001) / (union + .001)
    return round((dice_score), 2)



# NUOVA PARTE RICCARDO

def intLabel2String(label):
    if label == 0:
        return "B"
    elif label == 1:
        return "M"
    else:
        return "N"

def evaluateIOU(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    iou_score = (np.sum(intersection) + 1.) / (np.sum(union) + 1.)
    return round(iou_score, 2)

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

def evaluateDice(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.sum(prediction) + np.sum(label)
    dice_score = ((2 * np.sum(intersection)) + .001) / (union + .001)
    return round((dice_score), 2)

"""def evaluateMatthewsCorrelationCoefficient(y_pred, y_true, epsilon=1e-5):
    
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
    return mcc#tf.math.sqrt(tpr*ppv)"""


def customModelScore(iou, dice, tpr, ppv):
    return ((((iou + dice) / 2) - (dice - iou)) + tpr + ppv) / 3
def customModelScore2(iou, dice, tpr, ppv):
    return ((((iou + dice) / 2) - (dice - iou)) + (tpr * ppv)) / 2
def evaluateGeometricMean(tpr, ppv):
    return sqrt(tpr*ppv)

def evaluateSegmentationBM(ys_true, yc_true, y_pred, t=0.5):
    iou_scores = []
    dice_scores = []
    detection_pred = []
    detection_truth = []

    for truth, label, pred in zip(ys_true, yc_true, y_pred):

        if label[2] == 1:
            continue
        pred[pred >= t] = 1
        pred[pred < t] = 0

        curr_IOU = evaluateIOU(pred, truth)
        curr_DICE = evaluateDice(pred, truth)

        iou_scores.append(curr_IOU)
        dice_scores.append(curr_DICE)

        if np.count_nonzero(pred) == 0:
            detection_pred.append(0)
        else:
            if curr_IOU > 0.2:
                detection_pred.append(1)
            else:
                detection_pred.append(0)

        if np.count_nonzero(truth) == 0:
            detection_truth.append(0)
        else:
            detection_truth.append(1)

    detection_accuracy = metrics.accuracy_score(detection_truth, detection_pred)
    detection_bal_acc = metrics.balanced_accuracy_score(detection_truth, detection_pred)

    return iou_scores, dice_scores, detection_accuracy

def evaluateSegmentationRGB(y_true, y_pred, name="unknown", t=0.5, verbose=True):
    y_true = y_true.copy()
    y_pred = y_pred.copy()

    Pred_B_Eval = evaluateSegmentation(y_true[:, :, :, 0], y_pred[:, :, :, 0], t=t)
    Pred_M_Eval = evaluateSegmentation(y_true[:, :, :, 1], y_pred[:, :, :, 1], t=t)
    Pred_L_Eval = evaluateSegmentation(y_true[:, :, :, 2], y_pred[:, :, :, 2], t=t)

    if np.count_nonzero(y_pred[:, :, :, 2]) > 0:
        Pred_ProbBenign = genClassificationByMask(y_pred[:, :, :, 2], y_pred[:, :, :, 0])
        Pred_ProbMalign = genClassificationByMask(y_pred[:, :, :, 2], y_pred[:, :, :, 1])
        probVector = [Pred_ProbBenign, Pred_ProbMalign, 0]
    else:
        probVector = [0,0,1]

    if verbose:
        splitCount = {
            'Classes': ['IoU', 'Dice', 'TPR', 'TNR', ],
            'Benign': [round(np.mean(Pred_B_Eval["iou"]),3), round(np.mean(Pred_B_Eval["dice"]),3),
                       Pred_B_Eval["tpr"],Pred_B_Eval["tnr"]],
            'Malignant': [round(np.mean(Pred_M_Eval["iou"]),3), round(np.mean(Pred_M_Eval["dice"]),3),
                       Pred_M_Eval["tpr"],Pred_M_Eval["tnr"]],
            'Lesion': [round(np.mean(Pred_L_Eval["iou"]),3), round(np.mean(Pred_L_Eval["dice"]),3),
                       Pred_L_Eval["tpr"],Pred_L_Eval["tnr"]],
        }
        df = pd.DataFrame(data=splitCount, index=None)
        print(f"{name} | Evaluation Results:")
        print(df)
    return {"B":Pred_B_Eval, "M":Pred_M_Eval, "L":Pred_L_Eval, "classification":probVector}

def genClassificationByMask(y_true, y_pred):
    iou_scores = []
    dice_scores = []
    for truth, pred in zip(y_true, y_pred):
        curr_IOU = evaluateIOU(pred, truth)
        curr_DICE = evaluateDice(pred, truth)

        iou_scores.append(curr_IOU)
        dice_scores.append(curr_DICE)

    return iou_scores

def evaluateSegmentation(y_true, y_pred, t=0.5, det_t=0.5):
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

        curr_IOU = evaluateIOU(pred, truth)
        curr_DICE = evaluateDice(pred, truth)
        curr_intersection = evaluateTruePositiveRate(pred, truth)
        curr_tnr = evaluateTrueNegativeRate(pred, truth)
        curr_falsepositive = evaluateFalseDiscoveryRate(pred,truth)
        curr_mcc = 0#evaluateMatthewsCorrelationCoefficient(pred, truth)#evaluateF1Score(pred,truth)#0#
        curr_fmi = evaluateFowlkesMallowsIndex(pred,truth)

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

def confMatrixGenLabels(truths, predictions):
    y_pred = []
    y_truth = []

    for pred, truth in zip(predictions, truths):

        max_index_pred = pred.tolist().index(max(pred))
        max_index_truth = truth.tolist().index(max(truth))

        if max_index_pred == 0: y_pred.append("benign")
        if max_index_pred == 1: y_pred.append("malignant")
        if max_index_pred == 2: y_pred.append("normal")

        if max_index_truth == 0: y_truth.append("benign")
        if max_index_truth == 1: y_truth.append("malignant")
        if max_index_truth == 2: y_truth.append("normal")

    return y_truth, y_pred


def classification2Class(y_true, y_pred, verbose=False, modelName="unknown"):
    count_p = [0, 0]
    count_t = [0, 0]

    decoded_y_true = []
    decoded_y_pred = []

    correctPred = 0
    errorNaN = False
    if y_pred is not None:
        for pred, truth in zip(y_pred, y_true):
            if np.isnan(max(pred)):
                max_index_pred = 0
                errorNaN = True
            else:
                max_index_pred = pred.tolist().index(max(pred))

            max_index_truth = truth.tolist().index(max(truth))
            if (max_index_pred == max_index_truth):
                correctPred += 1

            decoded_y_true.append(max_index_truth)
            decoded_y_pred.append(max_index_pred)

            count_p[max_index_pred] += 1
            count_t[max_index_truth] += 1

        if errorNaN: print("ERROR: NaN values detected on predictions probability")
        matrix = confusion_matrix(decoded_y_true, decoded_y_pred)
        matrix = np.array(matrix).transpose()
        df_val = {'0': matrix[0], '1': matrix[1]}
        dataFrame_confusionMatrix = pd.DataFrame(data=df_val)

        accuracy = metrics.accuracy_score(decoded_y_true, decoded_y_pred)  # round((correctPred / y_true.shape[0]), 4)
        bal_acc = metrics.balanced_accuracy_score(decoded_y_true, decoded_y_pred)
        precision = metrics.precision_score(decoded_y_true, decoded_y_pred, average="weighted")
        recall = metrics.recall_score(decoded_y_true, decoded_y_pred, average="weighted")
        f1score = metrics.f1_score(decoded_y_true, decoded_y_pred, average="weighted")

        if verbose:
            print(f"{modelName} | Count Predictions : " + str(count_p))
            print(f"{modelName} | Count Real Values : " + str(count_t))
            print(f"{modelName} | Accuracy: " + str(accuracy) + "%")
            print(f"{modelName} | Confusion Matrix: ")
            print(dataFrame_confusionMatrix)
            #confusionMatrix.pretty_plot_confusion_matrix(dataFrame_confusionMatrix, pred_val_axis="y", modelName="ResNet50")
            print('-' * 50)

        return {"acc": accuracy, "bacc": bal_acc, "pre": precision, "rec": recall, "f1s": f1score,
                "dfConfMatrix": dataFrame_confusionMatrix, "countPreds": count_p, "countTruth": count_t}

def evaluateClassificationBMbySegm(y_true, y_pred, y_pred_GT, verbose=False, modelName="unknown"):
    count_p = [0, 0, 0]
    count_t = [0, 0, 0]
    correctPred = 0

    decoded_y_true = []
    decoded_y_pred = []

    errorNaN = False
    if y_pred is not None:
        for pred, predGT, truth in zip(y_pred, y_pred_GT, y_true):
            """if np.isnan(max(pred)):
                max_index_pred = 0
                errorNaN = True
            else:"""
            max_index_pred = pred.tolist().index(max(pred))
            max_index_truth = truth.tolist().index(max(truth))

            if max_index_truth != 2 and np.count_nonzero(predGT) != 0:
                decoded_y_true.append(max_index_truth)
                decoded_y_pred.append(max_index_pred)
                if (max_index_pred == max_index_truth):
                    correctPred += 1

                count_p[max_index_pred] += 1
                count_t[max_index_truth] += 1


    if errorNaN: print("ERROR: NaN values detected on predictions probability")

    matrix = confusion_matrix(decoded_y_pred, decoded_y_true)
    matrix = np.array(matrix).transpose()
    df_val = {'B': matrix[0], 'M': matrix[1]}
    dataFrame_confusionMatrix = pd.DataFrame(data=df_val)

    accuracy = metrics.accuracy_score(decoded_y_true, decoded_y_pred)#round((correctPred / y_true.shape[0]), 4)
    bal_acc = metrics.balanced_accuracy_score(decoded_y_true, decoded_y_pred)
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

    return {"acc": accuracy, "bacc":bal_acc, "pre": precision, "rec": recall, "f1s": f1score, "decoded_y_pred": decoded_y_pred,
            "dfConfMatrix": dataFrame_confusionMatrix, "countPreds": count_p, "countTruth": count_t}

def evalClassificationFromGT(y_true, y_pred, y_pred_GT, verbose=False, modelName="unknown"):
    count_p = [0, 0, 0]
    count_t = [0, 0, 0]
    correctPred = 0

    decoded_y_true = []
    decoded_y_pred = []

    errorNaN = False
    if y_pred is not None:
        for pred, predGT, truth in zip(y_pred, y_pred_GT, y_true):

            if np.isnan(max(pred)):
                max_index_pred = 0
                errorNaN = True
            else:
                # If the predicted mask is empty, set the label to "Normal"
                if np.count_nonzero(predGT) == 0:
                    max_index_pred = 2 #Normal
                # otherwise, get prediction from Benign-vs-Malignant-By-Ground-Truth models
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

    matrix = confusion_matrix(decoded_y_pred, decoded_y_true)
    matrix = np.array(matrix).transpose()
    df_val = {'B': matrix[0], 'M': matrix[1], 'N': matrix[2]}
    dataFrame_confusionMatrix = pd.DataFrame(data=df_val)

    accuracy = metrics.accuracy_score(decoded_y_true, decoded_y_pred)#round((correctPred / y_true.shape[0]), 4)
    bal_acc = metrics.balanced_accuracy_score(decoded_y_true, decoded_y_pred)
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

    return {"acc": accuracy, "bacc": bal_acc, "pre": precision, "rec": recall, "f1s": f1score, "ensemble_pred": decoded_y_pred,
            "dfConfMatrix": dataFrame_confusionMatrix, "countPreds": count_p, "countTruth": count_t}

def classification3Class(y_true, y_pred, n_classes, verbose=False, modelName="unknown"):

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

def evaluateClassificationHierarchy(y_true, y_pred):
    wrongPreds = 0
    errorNaN = False
    if y_pred is not None:
        for pred, truth in zip(y_pred, y_true):
            if np.isnan(max(pred)):
                max_index_pred = 0
                errorNaN = True
            else:
                max_index_pred = pred.tolist().index(max(pred))

            max_index_truth = truth.tolist().index(max(truth))
            if (max_index_pred != max_index_truth & max_index_truth == 2):
                wrongPreds += 2
            elif (max_index_pred != max_index_truth):
                wrongPreds += 1


    if errorNaN: print("ERROR: NaN values detected on predictions probability")
    accuracy = round(((y_true.shape[0]-wrongPreds) / y_true.shape[0]), 3)
    accuracy2 = round((((2*y_true.shape[0]) - wrongPreds) / (2*y_true.shape[0])), 3)

    return {"acc1": accuracy,"acc2": accuracy2, }
