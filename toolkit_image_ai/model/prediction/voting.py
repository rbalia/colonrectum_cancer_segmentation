import numpy as np

def segmentation_softVoting(pred_list, t=0.5, samples_axis=0):
    #Apply voting by mean (soft)
    voting_raw = np.mean(pred_list, axis=samples_axis)

    # Apply threshold
    voting = np.copy(voting_raw)
    voting[voting >= t] = 1.
    voting[voting < t] = 0

    return voting, voting_raw

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