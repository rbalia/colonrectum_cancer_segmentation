from sklearn.model_selection import KFold

def getFoldIdx(n_split, set):
    kf = KFold(n_splits=n_split)
    kf.get_n_splits(set)
    foldIdx = []
    for train_index, test_index in kf.split(set):
        foldIdx.append([train_index, test_index])
    return foldIdx