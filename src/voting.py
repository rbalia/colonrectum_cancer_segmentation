
from skimage import img_as_ubyte, img_as_float, io

from src import evaluate, preprocessing
from src import configs as conf
import numpy as np
from matplotlib import pyplot as plt

from src import plotter
def voting_gt_roi(roi_preds, gt_preds):
    voting_preds = []
    i = 0
    for gt_pred, roi_pred in zip(gt_preds, roi_preds):
        voting = [(gt_pred[0]+roi_pred[0])/2,
                  (gt_pred[1]+roi_pred[1])/2,
                  gt_pred[2]]
        voting_preds.append(voting)
        print(f"gt:{gt_pred} - roi:{roi_pred} - vot:{voting}")
    voting_preds = np.array(voting_preds)
    return voting_preds

def combine_roi_mask(roi_preds, masks):
    voting_preds = []
    for y_roi, mask in zip(roi_preds, masks):
        if np.count_nonzero(mask) > 0: # if mask is not empty
            B_prob = y_roi[0] * 0.66
            M_prob = y_roi[1] * 0.66
            N_prob = 1. - (B_prob + M_prob)
        else:
            B_prob = y_roi[0] / 3
            M_prob = y_roi[1] / 3
            N_prob = 1. - (B_prob + M_prob)
        voting = [B_prob,M_prob,N_prob]
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds

def combine_roi_mask_raw(roi_preds, masks):
    voting_preds = []
    for y_roi, mask in zip(roi_preds, masks):
        if np.count_nonzero(mask) > 0: # if mask is not empty
            B_prob = y_roi[0]
            M_prob = y_roi[1]
            N_prob = 0.
        else:
            B_prob = 0
            M_prob = 0
            N_prob = 1.
        voting = [B_prob,M_prob,N_prob]
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds

def combine_gt_roi(gt_preds, roi_preds, mask_pred):
    voting_preds = []
    i = 0
    for gt_pred, roi_pred, mask in zip(gt_preds, roi_preds, mask_pred):
        if np.count_nonzero(mask) > 0:  # if mask is not empty
            B_prob = (gt_pred[0]+roi_pred[0])/2
            M_prob = (gt_pred[1]+roi_pred[1])/2
            N_prob = 0.
        else:
            B_prob = 0.25
            M_prob = 0.25
            N_prob = 0.5
        voting = [B_prob, M_prob, N_prob]
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)


    return voting_preds

def classification_binary_softVoting(voter1, voter2):
    voting_preds = []
    for probVector1, probVector2 in zip(voter1, voter2):
        voting = (probVector1 + probVector2)/2
        print(voting)
        #voting = [round(voting[0],5), round(voting[1],5), round(voting[2],5)]
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds

def classification_testApproach(zipped_preds, weights=None):
    voting_preds = []
    for preds in zipped_preds:
        n_voters = len(preds)
        voting = None
        i = 0
        if weights is None:
            weights = np.repeat(1, n_voters, axis=0)
        for probVector in preds:
            if voting is None:
                voting = (probVector*weights[i] * max(probVector)) / n_voters # * max
                #voting = (probVector / max(probVector)) / n_voters
            else:
                voting += (probVector*weights[i] * max(probVector)) / n_voters
                #voting += (probVector / max(probVector)) / n_voters
        #voting = [round(voting[0],5), round(voting[1],5), round(voting[2],5)]
        voting_preds.append(voting)
        i+=1
    voting_preds = np.array(voting_preds)

    return voting_preds

def classification_softVoting(zipped_preds, weights=None):
    voting_preds = []
    for preds in zipped_preds:
        n_voters = len(preds)
        voting = None
        i = 0
        if weights is None:
            weights = np.repeat(1, n_voters, axis=0)
        for probVector in preds:
            if voting is None:
                voting = (probVector*weights[i]) / n_voters
            else:
                voting += (probVector*weights[i]) / n_voters
        #voting = [round(voting[0],5), round(voting[1],5), round(voting[2],5)]
            i += 1
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)
    return voting_preds

def classification_finalEnsamble(mc_pred, roi_pred,gt_pred, weights=None):
    voting_preds = []
    for mc, roi,gt in zip(mc_pred, roi_pred, gt_pred):

        """roi *= np.max(roi)
        mc *= np.max(mc)
        gt *= np.max(gt)"""

        if np.max(mc) > 0.9:
            mc = mc.copy() * 1

        Bp = (mc[0] + roi[0] + gt[0])/3
        Mp = (mc[1] + roi[1] + gt[1])/3
        Np = (mc[2] + roi[2]/2 + gt[2]/2)/3
        voting = [Bp, Mp, Np]
        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds


def classification_hardVoting(zipped_preds):
    voting_preds = []
    for preds in zipped_preds:
        n_voters = len(preds)
        voting = None
        for probVector in preds:
            n_class = len(probVector)
            if voting is None:
                voting = np.repeat(0.0, n_class, axis=0)

            label = probVector.tolist().index(max(probVector))
            voting[label] += 1/n_voters

        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds

def classification_normVoting(zipped_preds, weights=None):
    voting_preds = []
    for preds in zipped_preds:
        i=0
        n_voters = len(preds)
        voting = None
        if weights is None:
            weights = np.repeat(1, n_voters, axis=0)
        for probVector in preds:
            n_class = len(probVector)
            if voting is None:
                voting = np.repeat(0.0, n_class, axis=0)

            #label = probVector.tolist().index(max(probVector))
            voting += ((probVector*weights[i]) / np.max(probVector)) / n_voters
            i += 1


        voting_preds.append(voting)
    voting_preds = np.array(voting_preds)

    return voting_preds


    """hardVotingPreds = np.repeat(np.array([[0.0, 0.0, 0.0]]), [YC_test.numpy().shape[0]], axis=0)
    i = 0
    
    for preds in zip(resnet50_preds, inceptionv3_preds, xception_preds, inceptionresnetv2_preds,
                     densenet201_preds):
        for probVector in preds:
            label = probVector.tolist().index(max(probVector))
            hardVotingPreds[i][label] += 0.2
        i += 1"""

def segmentation_MinimumVoting(zipped_preds, t=0.5):
    voting_preds = []
    voting_preds_raw = []

    for preds in zipped_preds:
        voting = None
        for pred in preds:
            if voting is None:
                voting = pred#/np.max(pred)
            else:
                voting = np.minimum(voting, pred)#/np.max(pred))

        voting_preds_raw.append(np.copy(voting))

        voting[voting >= t] = 1.
        voting[voting < t] = 0

        voting_preds.append(voting)

    voting_preds = np.array(voting_preds)
    voting_preds_raw = np.array(voting_preds_raw)

    return voting_preds, voting_preds_raw

def segmentation_MaximumVoting(zipped_preds, t=0.5):
    voting_preds = []
    voting_preds_raw = []

    for preds in zipped_preds:
        voting = None
        for pred in preds:
            if voting is None:
                voting = pred#/np.max(pred)
            else:
                voting = np.maximum(voting, pred)#/np.max(pred))

        voting_preds_raw.append(np.copy(voting))

        voting[voting >= t] = 1.
        voting[voting < t] = 0

        voting_preds.append(voting)

    voting_preds = np.array(voting_preds)
    voting_preds_raw = np.array(voting_preds_raw)

    return voting_preds, voting_preds_raw

def segmentation_SoftVoting(zipped_preds, t=0.5, returnStats=False, applyAdaptive=False):
    validMaskCnt_list = []
    thresholds_list = []
    voting_preds = []
    voting_preds_raw = []
    validMaskCnt = 0

    for preds in zipped_preds:
        voting=None
        validMaskCnt = 0
        n_voters = len(preds)
        for pred in preds:
            if voting is None:
                voting = pred / n_voters
            else:
                voting += pred / n_voters

            if np.max(pred) >= t:
                validMaskCnt += 1

        voting_raw = np.copy(voting)
        voting_preds_raw.append(np.copy(voting))

        #th = (t * 40 + np.max(voting)) / 41
        #th += ((validMaskCnt / n_voters) - 0.5) / 1000
        th = t
        if applyAdaptive:
            th += (np.max(voting) - (1-t)) / 100
            th += ((validMaskCnt / n_voters) - (1-t)) / 100
            th = 1 - th

        voting[voting >= th] = 1.
        voting[voting < th] = 0

        voting_preds.append(voting)
        validMaskCnt_list.append(validMaskCnt)
        thresholds_list.append(th)

    voting_preds = np.array(voting_preds)
    voting_preds_raw = np.array(voting_preds_raw)

    if returnStats:
        return voting_preds, voting_preds_raw, {"validMaskCnt":validMaskCnt_list, "threshold": thresholds_list}
    else:
        return voting_preds, voting_preds_raw

def segmentation_SoftVoting(zipped_preds, t=0.5, returnStats=False, applyAdaptive=False):
    validMaskCnt_list = []
    thresholds_list = []
    voting_preds = []
    voting_preds_raw = []
    validMaskCnt = 0

    for preds in zipped_preds:
        voting=None
        validMaskCnt = 0
        n_voters = len(preds)
        for pred in preds:
            if voting is None:
                voting = pred / n_voters
            else:
                voting += pred / n_voters

            if np.max(pred) >= t:
                validMaskCnt += 1

        voting_raw = np.copy(voting)
        voting_preds_raw.append(np.copy(voting))

        #th = (t * 40 + np.max(voting)) / 41
        #th += ((validMaskCnt / n_voters) - 0.5) / 1000
        th = t
        if applyAdaptive:
            th += (np.max(voting) - (1-t)) / 100
            th += ((validMaskCnt / n_voters) - (1-t)) / 100
            th = 1 - th

        voting[voting >= th] = 1.
        voting[voting < th] = 0

        voting_preds.append(voting)
        validMaskCnt_list.append(validMaskCnt)
        thresholds_list.append(th)

    voting_preds = np.array(voting_preds)
    voting_preds_raw = np.array(voting_preds_raw)

    if returnStats:
        return voting_preds, voting_preds_raw, {"validMaskCnt":validMaskCnt_list, "threshold": thresholds_list}
    else:
        return voting_preds, voting_preds_raw

def segmentation_SoftVoting2(pred_list, th=0.5):

    n_voters = len(pred_list)
    voting_raw = pred_list[0]/n_voters
    for pred in pred_list[1:]:
        voting_raw += pred / n_voters

    #voting_raw = np.mean(pred_list, axis=0)

    voting = np.copy(voting_raw)
    voting[voting >= th] = 1.
    voting[voting < th] = 0

    return voting, voting_raw


def segmentation_SoftVoting3(pred_list, t=0.5, samples_axis=0):

    voting_raw = np.mean(pred_list, axis=samples_axis)

    voting = np.copy(voting_raw)
    voting[voting >= t] = 1.
    voting[voting < t] = 0

    return voting, voting_raw

def classificationFromSegmentation_Dice(predictions, t=0.5):

    probVector = []
    predictions = preprocessing.thresholdMaskSet(predictions, t)

    for lesion, benign, malignant in zip(predictions[:, :, :, 2], predictions[:, :, :, 0], predictions[:, :, :, 1]):

        if np.count_nonzero(lesion) > 0:
            dice_benign = evaluate.evaluateDice(benign, lesion)
            dice_malignant = evaluate.evaluateDice(malignant, lesion)

            benign_prob = (dice_benign + 0.01) / (dice_benign + dice_malignant + 0.01 )
            malignant_prob = (dice_malignant + 0.01) / (dice_benign + dice_malignant + 0.01)

            probVector.append([benign_prob, malignant_prob, 0])
        else:
            probVector.append([0, 0, 1])
    probVector = np.array(probVector)
    return probVector

def classificationFromSegmentation_Sum(predictions, t=0.5):


    probVector = []

    lesions_set = np.copy(predictions[:, :, :, 2])
    lesions_set = preprocessing.thresholdMaskSet(lesions_set, t)

    benign_set = np.copy(predictions[:, :, :, 0])
    malignant_set = np.copy(predictions[:, :, :, 1])



    for lesion, benign, malignant in zip(lesions_set, benign_set, malignant_set):
        nonZeroPixelsCount = np.count_nonzero(lesion)
        if nonZeroPixelsCount > 0:
            benign[lesion < 1] = 0
            malignant[lesion < 1] = 0

            benign_roiSum = np.sum(benign)/nonZeroPixelsCount
            malignant_roiSum = np.sum(malignant)/nonZeroPixelsCount

            benign_prob = benign_roiSum / (benign_roiSum+malignant_roiSum)
            malignant_prob = malignant_roiSum / (benign_roiSum+malignant_roiSum)

            probVector.append([benign_prob, malignant_prob, 0])

            #plotter.printBrief3Cells("title", ["Lesion", f"{benign_prob}", f"{malignant_prob}"], [lesion, benign, malignant])
        else:
            probVector.append([0, 0, 1])
    probVector = np.array(probVector)
    return probVector

def segmentation_combinationProbBased2(LN, LNraw, BM, BM_raw,BMN, BMN_raw, validLNcount_list, validBMcount_list, imgs):
    combination_voting = []
    combination_voting_raw = []

    i = 0
    for ln, ln_raw, bm, bm_raw,bmn,bmn_raw, validLNcount, validBMcount, img in zip(LN, LNraw,
                                                                                   BM, BM_raw,
                                                                                   BMN, BMN_raw,validLNcount_list,
                                                                                   validBMcount_list, imgs):

        msk = ln.copy()
        msk[bm == 1] = 1

        if np.count_nonzero(ln) > 0:
            ln_roi = ln_raw.copy()
            ln_roi[ln == 0] = 0
            ln_nonzero_pixels = np.nonzero(ln_roi)
            ln_avg = ln_roi[ln_nonzero_pixels].mean()

            bm_roi = bm_raw.copy()
            bm_roi[bm == 0] = 0
            bm_nonzero_pixels = np.nonzero(bm_roi)
            bm_avg = bm_roi[bm_nonzero_pixels].mean()

            if ln_avg > bm_avg:
                combination_voting.append(ln)
                combination_voting_raw.append(ln_raw)
            else:
                combination_voting.append(bm)
                combination_voting_raw.append(bm_raw)

            """utils_print.printBrief4Cells("title", ["img",f"ln {ln_avg}",f"bm {bm_avg}", "chosen"], [img, ln_roi, bm_roi,
                                         combination_voting[-1]])"""
        else:
            #if np.count_nonzero(bmn) > 0 and np.count_nonzero(bm) > 0 and validLNcount >= 5 and validBMcount >= 5:
            #    combination_voting.append(bm)
            #    combination_voting_raw.append(bm_raw)
            #else:
            combination_voting.append(ln)
            combination_voting_raw.append(ln_raw)



    combination_voting = np.array(combination_voting)
    combination_voting_raw = np.array(combination_voting_raw)

    return combination_voting, combination_voting_raw

def segmentation_combinationProbBased3(LN, LNraw, BM, BM_raw,BMN, BMN_raw, validLNcount_list, validBMcount_list, imgs):
    combination_voting = []
    combination_voting_raw = []

    i = 0
    for ln, ln_raw, bm, bm_raw,bmn,bmn_raw, validLNcount, validBMcount, img in zip(LN, LNraw,
                                                                                   BM, BM_raw,
                                                                                   BMN, BMN_raw,validLNcount_list,
                                                                                   validBMcount_list, imgs):

        msk = ln.copy()
        msk[bm == 1] = 1

        #print(f"{validLNcount} | {validBMcount}" )

        if np.count_nonzero(ln) > 0 and np.count_nonzero(bm):
            ln_roi = ln_raw.copy()
            ln_roi[ln == 0] = 0
            ln_nonzero_pixels = np.nonzero(ln_roi)
            ln_avg = ln_roi[ln_nonzero_pixels].mean()

            bm_roi = bm_raw.copy()
            bm_roi[bm == 0] = 0
            bm_nonzero_pixels = np.nonzero(bm_roi)
            bm_avg = bm_roi[bm_nonzero_pixels].mean()

            imgln_roi= 1-img.copy()
            imgln_roi[ln == 0] = 0
            imgln_nonzero_pixels = np.nonzero(imgln_roi)
            imgln_avg = imgln_roi[imgln_nonzero_pixels].mean()

            imgbm_roi = 1-bm_raw.copy()
            imgbm_roi[bm == 0] = 0
            imgbm_nonzero_pixels = np.nonzero(imgbm_roi)
            imgbm_avg = imgbm_roi[imgbm_nonzero_pixels].mean()

            if imgln_avg > imgbm_avg:
                combination_voting.append(ln)
                combination_voting_raw.append(ln_raw)
            #elif validLNcount < validBMcount and imgln_avg > imgbm_avg:
            #    combination_voting.append(ln)
            #    combination_voting_raw.append(ln_raw)
            else:
                combination_voting.append(bm)
                combination_voting_raw.append(bm_raw)
            """utils_print.printBrief4Cells("title", ["img",f"ln {ln_avg}",f"bm {bm_avg}", "chosen"], [img, ln_roi, bm_roi,
                                         combination_voting[-1]])"""
        else:
            #if np.count_nonzero(bmn) > 0 and np.count_nonzero(bm) > 0 and validLNcount >= 5 and validBMcount >= 5:
            #    combination_voting.append(bm)
            #    combination_voting_raw.append(bm_raw)
            #else:
            combination_voting.append(ln)
            combination_voting_raw.append(ln_raw)



    combination_voting = np.array(combination_voting)
    combination_voting_raw = np.array(combination_voting_raw)

    return combination_voting, combination_voting_raw

def segmentation_combinationProbBased(LN, LNraw, BM, BM_raw):
    combination_voting = []
    combination_voting_raw = []

    i = 0
    for ln, ln_raw, bm, bm_raw in zip(LN, LNraw, BM, BM_raw):

        if np.count_nonzero(ln) > 0:
            ln_roi = ln_raw.copy()
            ln_roi[ln == 0] = 0
            ln_nonzero_pixels = np.nonzero(ln_roi)
            ln_avg = ln_roi[ln_nonzero_pixels].mean()

            bm_roi = bm_raw.copy()
            bm_roi[bm == 0] = 0
            bm_nonzero_pixels = np.nonzero(bm_roi)
            bm_avg = bm_roi[bm_nonzero_pixels].mean()

            if ln_avg > bm_avg:
                combination_voting.append(ln)
                combination_voting_raw.append(ln_raw)
            else:
                combination_voting.append(bm)
                combination_voting_raw.append(bm_raw)
        else:
            combination_voting.append(ln)
            combination_voting_raw.append(ln_raw)

    combination_voting = np.array(combination_voting)
    combination_voting_raw = np.array(combination_voting_raw)

    return combination_voting, combination_voting_raw


def segmentation_voting_nonEmpyVoters(zipped_preds, t=0.5):
    voting_preds = []
    voting_preds_raw = []

    for preds in zipped_preds:
        voting = None
        n_voters = 0

        for pred in preds:
            pred_thresholded = pred.copy()
            pred_thresholded[pred_thresholded >= t] = 1.
            pred_thresholded[pred_thresholded < t] = 0.
            if np.count_nonzero(pred_thresholded) > 0:
                n_voters += 1
            if voting is None:
                voting = pred
            else:
                voting += pred

        if n_voters > 0:
            voting /= n_voters
        voting_preds_raw.append(np.copy(voting))

        voting[voting >= t] = 1.
        voting[voting < t] = 0
        voting_preds.append(voting)

    return voting_preds, voting_preds_raw


def segmentation_voting_2(zipped_preds, n_voters, t=0.5):
    print('-' * conf.dlw)
    print('Generate Segmentation Voting...')
    print('-' * conf.dlw)
    print(f"Number of Voters: {n_voters}")

    voting_preds = []
    voting_preds_raw = []

    for preds in zipped_preds:
        voting = None

        weights = []
        for pred in preds:
            weights.append(np.max(pred))
        weights = np.array(weights)
        weights -= np.min(weights)
        weights /= np.max(weights)

        for pred, w in zip(preds, weights):
            if voting is None:
                voting = (pred*w) / n_voters
            else:
                voting += (pred*w) / n_voters

        voting_preds_raw.append(np.copy(voting))

        voting[voting >= t] = 1.
        voting[voting < t] = 0
        voting_preds.append(voting)

    return voting_preds, voting_preds_raw


