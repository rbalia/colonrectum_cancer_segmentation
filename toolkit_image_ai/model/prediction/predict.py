from toolkit_image_ai.model.prediction import voting
import numpy as np

def test_time_augmentation_prediction(X_test, mdl):
    X_test_flip0 = np.flip(X_test.copy(), 0)
    X_test_flip1 = np.flip(X_test.copy(), 1)
    X_test_flip2 = np.flip(X_test.copy(), -1)

    ib_pred = mdl.predict(X_test, verbose=False)
    ib_pred_flip0 = mdl.predict(X_test_flip0, verbose=False)
    ib_pred_flip1 = mdl.predict(X_test_flip1, verbose=False)
    ib_pred_flip2 = mdl.predict(X_test_flip2, verbose=False)

    ib_pred_flip0 = np.flip(ib_pred_flip0, 0)
    ib_pred_flip1 = np.flip(ib_pred_flip1, 1)
    ib_pred_flip2 = np.flip(ib_pred_flip2, -1)

    predictions = [ib_pred, ib_pred_flip0, ib_pred_flip1, ib_pred_flip2]

    ib_augVoting_soft, ib_augVoting_soft_raw = \
        voting.segmentation_softVoting(predictions, t=0.5)

    del ib_pred
    del ib_pred_flip0
    del ib_pred_flip1
    del ib_pred_flip2

    return ib_augVoting_soft, ib_augVoting_soft_raw


def predict_on_batch(model, x, batch_size=8):
    preds = []
    for i in range(0, len(x), batch_size):
        batch_x = x[i:i + batch_size]
        batch_preds = model.predict(batch_x, verbose=False, batch_size=batch_size)
        preds.append(batch_preds)
    return np.concatenate(preds, axis=0)

def predict_on_batch_from_generator(model, datagenerator):
    preds = []
    for batch_id, batch in enumerate(datagenerator):
        batch_preds = model.predict(batch[0], verbose=False)
        preds.append(batch_preds)
    return np.concatenate(preds, axis=0)