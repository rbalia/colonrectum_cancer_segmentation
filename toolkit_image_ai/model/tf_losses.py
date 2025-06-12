from toolkit_image_ai.model.tf_metrics import iou_coef, dice_coef, gMean

def iou_coef_loss(y_true, y_pred):
    return 1 - iou_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def gMean_loss(y_true, y_pred):
    return 1.0 - gMean(y_true, y_pred)