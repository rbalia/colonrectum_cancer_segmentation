from datetime import date

today = date.today()
todayStr = today.strftime("%b_%d_%Y")

smooth = 1.
dlw = 50

img_rows = 128 #480
img_cols = 128 #640
channels = 1

img_rows_hd = 590
img_cols_hd = 620

img_rows_proc = 480
img_cols_proc = 640

batch_size = 16
epochs = 100
learning_rate = 1e-4
shuffleState = None

pkgDir = "dataset/binaryPkg/"
setDir = "dataset/dataset_BUSI/"
mdlDir = "models/"
prdDir = "preds/"
