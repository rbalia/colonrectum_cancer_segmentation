from __future__ import print_function

import os
import json

import tensorflow as tf
import tensorflow.keras.backend as K
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
import time

from src import models_builder as modelBuilder, evaluate
from src import voting
from src.augmentation import flip

smooth = 1.
import glob
import re
import fnmatch
import numpy as np
import cv2 as cv
import skimage

from skimage.io import imsave, imread, imread_collection
from skimage.transform import resize

from skimage import io
from skimage import img_as_ubyte, img_as_float
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from src import configs as conf, plotter
import matplotlib.pyplot as plt

def selectModel(modelName, modelDir, fold):
    # Get models from given dir
    modelsList = os.listdir(modelDir)

    # Filter by Model Name
    byNameIterator = filter(lambda x: modelName in x, modelsList)
    byName = list(byNameIterator)

    # Filter by Fold
    byFoldIterator = filter(lambda x: f"fold0{fold}" in x, byName)

    # Sort for best selection (if metrics are present)
    byFold = list(byFoldIterator)
    byFold.sort(reverse=True)

    # Return the best model path
    modelPath = modelDir + byFold[0]

    return modelPath

def scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        return lr * 0.5
    else:
        return lr
def trainClassifier(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, Y_test, fold=99,
             saveModelFlag=False, classes=3):

    # Define compiling parameters
    #if classes == 2:
    #    loss = "binary_crossentropy"
    #else:
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    optimizer = "adam"#SGD(learning_rate=0.0001) #"sgd"
    epochs = 100
    batch = conf.batch_size

    # Define callbacks
    useCheckpoint = True
    checkpointPath = f"./{conf.mdlDir}checkpoints/checkpoint_multiclass.h5"
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
    mc = ModelCheckpoint(checkpointPath, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    if useCheckpoint:
        callbacks = [es, mc]
    else:
        callbacks = [es]

    # Start Training
    print(f'Start Training | {modelName}')
    # Build a frozen model
    #model = modelBuilder.editClassifier(model, classes, "softmax", n_trainableLayers=20)

    """# First, train the frozen model for 10 epochs
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=10,
                        validation_data=(X_val, Y_val))

    # Then, unfreeze the last N layers and train
    model = modelBuilder.freezeLayers(model, n_trainableLayers=20)"""
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
    time.sleep(3)

    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpointPath)# = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    acc = round(model.evaluate(X_test, Y_test)[1], 3)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-acc{acc}")

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds

def train_MC(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, Y_test, fold=99,
             saveModelFlag=False, classes=3):

    # Define compiling parameters
    if classes == 2:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    optimizer = "adam"#SGD(learning_rate=0.0001) #"sgd"
    epochs = 100
    batch = conf.batch_size

    # Define callbacks
    useCheckpoint = True
    checkpointPath = f"./{conf.mdlDir}checkpoints/checkpoint_multiclass.h5"
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
    mc = ModelCheckpoint(checkpointPath, monitor="val_accuracy", mode="max", verbose=1, save_best_only=True)
    if useCheckpoint:
        callbacks = [es, mc]
    else:
        callbacks = [es]

    # Start Training
    print(f'Start Training | {modelName}')
    # Build a frozen model
    model = modelBuilder.editClassifier(model, classes, "softmax", n_trainableLayers=20)

    """# First, train the frozen model for 10 epochs
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=10,
                        validation_data=(X_val, Y_val))

    # Then, unfreeze the last N layers and train
    model = modelBuilder.freezeLayers(model, n_trainableLayers=20)"""
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
    time.sleep(3)

    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpointPath)# = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    acc = round(model.evaluate(X_test, Y_test)[1], 3)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-acc{acc}")

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds

def saveCheckpoint(model, X, Y, experiment, modelName, fold, checkpoint_path, useCheckpoint, saveModelFlag):
    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpoint_path)  # = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    iou = round(model.evaluate(X, Y)[1], 3)
    if saveModelFlag:
        # model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}")

def train_SEG_AdvEval(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, fold=99, saveModelFlag=False,
              useCheckpoint=True, useLrScheduler=False):

    # Define compiling parameters
    #s = loss_functions.Semantic_loss_functions()
    #s.focal_tversky
    #loss = "binary_focal_crossentropy"#"binary_crossentropy"#tf.nn.sigmoid_cross_entropy_with_logits
    loss = "binary_crossentropy"
    optimizer = "adam"  #tf.keras.optimizers.Adam(learning_rate=0.0001) SGD(lr=0.0001, momentum=0.9)#"sgd" #tf.keras.optimizers.Adam() #"adam" #tfa.optimizers.AdamW(weight_decay=0.0001)#tf.keras.optimizers.Adam()# "#Adam(amsgrad=True)#"adam"
    #metrics = [tf.keras.metrics.BinaryIoU(threshold=0.5), iou_coef, dice_coef]  # [s.dice_coef, s.sensitivity, s.specificity]
    metrics = [iou_coef, dice_coef]
    epochs = 100
    #tf.keras.metrics.BinaryIoU(target_class_id=[0, 1], threshold=0.5),
    batch = conf.batch_size

    # Define callbacks
    checkpointPath_max_dice = f"./{conf.mdlDir}{experiment}/ckpt_max_dice.h5"
    checkpointPath_min_loss = f"./{conf.mdlDir}{experiment}/ckpt_min_loss.h5"
    history_Path = f"./{conf.mdlDir}{experiment}/history_{modelName}_fold{fold}"
    es = EarlyStopping(monitor='val_dice_coef', mode='max', verbose=1, patience=20)
    mc1 = ModelCheckpoint(checkpointPath_max_dice, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1)
    mc2 = ModelCheckpoint(checkpointPath_min_loss, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
    lrs = LearningRateScheduler(scheduler, verbose=1)
    callbacks = [es]
    if useCheckpoint:
        callbacks = callbacks + [mc1, mc2]
    if useLrScheduler:
        callbacks = callbacks + [lrs]

    # Start Training
    print(f'Start Training | {modelName}')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    try:
        history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
        time.sleep(3)
    except KeyboardInterrupt:

        saveCheckpoint(model, X_val, Y_val, experiment, modelName + "_maxDice", fold,
                       checkpointPath_max_dice, useCheckpoint, saveModelFlag)
        saveCheckpoint(model, X_val, Y_val, experiment, modelName + "_minLoss", fold,
                       checkpointPath_min_loss, useCheckpoint, saveModelFlag)
        json.dump(history.history, open(history_Path, 'w'))

    # Save Checkpoint or Model
    saveCheckpoint(model, X_val, Y_val, experiment, modelName + "_maxDice", fold,
                   checkpointPath_max_dice, useCheckpoint, saveModelFlag)
    saveCheckpoint(model, X_val, Y_val, experiment, modelName + "_minLoss", fold,
                   checkpointPath_min_loss, useCheckpoint, saveModelFlag)
    json.dump(history.history, open(history_Path, 'w'))

    tf.keras.backend.clear_session()
    print('-' * conf.dlw)

def train_SEG(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, fold=99, saveModelFlag=False,
              useCheckpoint=True, useLrScheduler=False):

    # Define compiling parameters
    #s = loss_functions.Semantic_loss_functions()
    #s.focal_tversky
    #loss = "binary_focal_crossentropy"#"binary_crossentropy"#tf.nn.sigmoid_cross_entropy_with_logits
    loss = "binary_crossentropy"#"mse"
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer ="adam"  #Adam(learning_rate=0.01) #tf.keras.optimizers.Adam(learning_rate=0.0001) SGD(lr=0.0001, momentum=0.9)#"sgd" #tf.keras.optimizers.Adam() #"adam" #tfa.optimizers.AdamW(weight_decay=0.0001)#tf.keras.optimizers.Adam()# "#Adam(amsgrad=True)#"adam"
    #metrics = [tf.keras.metrics.BinaryIoU(threshold=0.5), iou_coef, dice_coef]  # [s.dice_coef, s.sensitivity, s.specificity]
    metrics = [iou_coef, dice_coef]
    epochs = 100
    #tf.keras.metrics.BinaryIoU(target_class_id=[0, 1], threshold=0.5),
    batch = conf.batch_size

    # Define callbacks
    checkpointPath = f"./{conf.mdlDir}{experiment}/checkpoint.h5"
    es = EarlyStopping(monitor='val_dice_coef', mode='max', verbose=1, patience=20)
    mc = ModelCheckpoint(checkpointPath, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1)
    lrs = LearningRateScheduler(scheduler, verbose=1)
    callbacks = [es]
    if useCheckpoint:
        callbacks = callbacks + [mc]
    if useLrScheduler:
        callbacks = callbacks + [lrs]

    # Start Training
    print(f'Start Training | {modelName}')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    try:
        history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
        time.sleep(3)
    except KeyboardInterrupt:
        # Save Checkpoint or Model
        if useCheckpoint:
            model.load_weights(checkpointPath)  # = tf.keras.models.load_model(checkpointPath)
            time.sleep(3)
        iou = round(model.evaluate(X_val, Y_val)[1], 3)
        preds = model.predict(X_val)
        loss_eval = round(loss_fn(Y_val, preds).numpy(),5)
        #iou = evaluate.evaluateIOU(preds, Y_val)
        print(loss_eval)
        if saveModelFlag:
            # model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
            model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}-loss{loss_eval}")

    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpointPath)# = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    iou = round(model.evaluate(X_val, Y_val)[1], 3)
    preds = model.predict(X_val)
    loss_eval = round(loss_fn(Y_val, preds).numpy(),5)
    #iou = evaluate.evaluateIOU(preds, Y_val)
    print(loss_eval)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}-loss{loss_eval}")

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds

def train_2SEG(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, fold=99, saveModelFlag=False,
              useCheckpoint=True, useLrScheduler=False):

    # Define compiling parameters
    #s = loss_functions.Semantic_loss_functions()
    #s.focal_tversky
    loss = "binary_crossentropy"#tf.nn.sigmoid_cross_entropy_with_logits
    optimizer = "adam"  #tf.keras.optimizers.Adam(learning_rate=0.0001) SGD(lr=0.0001, momentum=0.9)#"sgd" #tf.keras.optimizers.Adam() #"adam" #tfa.optimizers.AdamW(weight_decay=0.0001)#tf.keras.optimizers.Adam()# "#Adam(amsgrad=True)#"adam"
    metrics = [iou_coef, dice_coef]  # [s.dice_coef, s.sensitivity, s.specificity]
    epochs = 100
    batch = conf.batch_size

    # Define callbacks
    checkpointPath = f"./{conf.mdlDir}{experiment}/checkpoint.h5"
    es = EarlyStopping(monitor='val_out1_dice_coef', mode='max', verbose=1, patience=20)
    mc = ModelCheckpoint(checkpointPath, monitor="val_out1_dice_coef", mode="max", save_best_only=True, verbose=1)
    lrs = LearningRateScheduler(scheduler, verbose=1)
    callbacks = [es]
    if useCheckpoint:
        callbacks = callbacks + [mc]
    if useLrScheduler:
        callbacks = callbacks + [lrs]

    # Start Training
    print(f'Start Training | {modelName}')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    try:
        history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
        time.sleep(3)
    except KeyboardInterrupt:
        # Save Checkpoint or Model
        if useCheckpoint:
            model.load_weights(checkpointPath)  # = tf.keras.models.load_model(checkpointPath)
            time.sleep(3)
        iou = round(model.evaluate(X_val, Y_val)[1], 3)
        if saveModelFlag:
            # model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
            model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}")

    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpointPath)# = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    iou = round(model.evaluate(X_val, Y_val)[1], 3)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}")

    # Predict
    preds1, preds2 = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds1, preds2

def train_SEG_CLA(model, modelName, experiment, X_train, Y_train, X_val, Y_val, X_test, fold=99, saveModelFlag=False):

    # Define compiling parameters
    #s = loss_functions.Semantic_loss_functions()
    #s.focal_tversky
    loss = {"seg_output":dice_coef_loss, "cla_output":"categorical_crossentropy"}#tf.nn.sigmoid_cross_entropy_with_logits
    optimizer = "adam"  # SGD(lr=0.0001, momentum=0.9)#"sgd" #tf.keras.optimizers.Adam() #"adam" #tfa.optimizers.AdamW(weight_decay=0.0001)#tf.keras.optimizers.Adam()# "#Adam(amsgrad=True)#"adam"
    metrics = {"seg_output":[iou_coef, dice_coef],"cla_output": ["accuracy"]}  # [s.dice_coef, s.sensitivity, s.specificity]
    epochs = 50
    batch = conf.batch_size

    # Define callbacks

    # Start Training
    print(f'Start Training | {modelName}')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    try:
        history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs, shuffle=False,
                        validation_data=(X_val, Y_val))
        time.sleep(3)
    except KeyboardInterrupt:
        # Save Checkpoint or Model
        iou = round(model.evaluate(X_val, Y_val)[1], 3)
        if saveModelFlag:
            # model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
            model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}")

    # Save Checkpoint or Model
    iou = round(model.evaluate(X_val, Y_val)[1], 3)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}{experiment}/{modelName}-fold{fold:02d}-iou{iou}")

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds


def train_SEG_BM(model, modelName, X_train, Y_train, X_val, Y_val, X_test, fold=99, saveModelFlag=False):

    # Define compiling parameters
    #s = loss_functions.Semantic_loss_functions()
    #s.focal_tversky
    loss = "binary_crossentropy"#tf.nn.sigmoid_cross_entropy_with_logits
    optimizer = "adam"  # SGD(lr=0.0001, momentum=0.9)#"sgd" #tf.keras.optimizers.Adam() #"adam" #tfa.optimizers.AdamW(weight_decay=0.0001)#tf.keras.optimizers.Adam()# "#Adam(amsgrad=True)#"adam"
    metrics = [iou_coef, dice_coef]  # [s.dice_coef, s.sensitivity, s.specificity]
    epochs = 100
    batch = conf.batch_size

    # Define callbacks
    useCheckpoint = True
    checkpointPath = f"./{conf.mdlDir}checkpoints/checkpoint_segmentation.h5"
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
    mc = ModelCheckpoint(checkpointPath, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1)
    if useCheckpoint:
        callbacks = [es, mc]
    else:
        callbacks = [es]

    # Start Training
    print(f'Start Training | {modelName}')
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(X_train, Y_train, verbose=1, batch_size=batch, epochs=epochs, shuffle=True,
                        validation_data=(X_val, Y_val), callbacks=callbacks)
    time.sleep(3)

    # Save Checkpoint or Model
    if useCheckpoint:
        model.load_weights(checkpointPath)# = tf.keras.models.load_model(checkpointPath)
        time.sleep(3)
    iou = round(model.evaluate(X_val, Y_val)[1], 3)
    if saveModelFlag:
        #model.save_weights(f"./models/multiclass/{modelName}-fold{fold:02d}-acc{acc}.h5", save_format="h5")
        model.save(f"./{conf.mdlDir}segmentationBM/{modelName}-fold{fold:02d}-iou{iou}")

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()
    print('-' * conf.dlw)
    return history, preds

def genAlteredSet(imgSet, param):
    imgSet_t = imgSet.copy()
    i= 0
    for img in imgSet:
        img = cv2.flip(img, param)
        if len(img.shape)<3:
            img = np.expand_dims(img, axis=-1)
        imgSet_t[i] = img
        i += 1
    return imgSet_t

def loadAndPredict_AvgAug(modelName, X_test, modelDir, fold, mode="cla"):
    # Get Trained Model path
    modelPath = selectModel(modelName, modelDir, fold)

    # Loading Trained Model
    print(f'Loading Model | {modelName} | {modelPath}')
    tf.keras.backend.clear_session()
    if mode == "seg":
        custom_objects = {"iou_coef": iou_coef, "dice_coef": dice_coef}
        model = tf.keras.models.load_model(modelPath, compile=False, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(modelPath, compile=False)

    # Predict
    preds = model.predict(X_test)

    X_test_1 = genAlteredSet(X_test, 0)
    X_test_2 = genAlteredSet(X_test, 1)
    X_test_3 = genAlteredSet(X_test, -1)

    pred1 = model.predict(X_test_1)
    pred1_t = genAlteredSet(pred1, 0)
    pred2 = model.predict(X_test_2)
    pred2_t = genAlteredSet(pred2, 1)
    pred3 = model.predict(X_test_3)
    pred3_t = genAlteredSet(pred3, -1)

    tf.keras.backend.clear_session()

    _, meanPred_raw = voting.segmentation_SoftVoting(zip(preds, pred1_t, pred2_t, pred3_t))
    _, maxPred_raw = voting.segmentation_MaximumVoting(zip(preds, pred1_t, pred2_t, pred3_t))

    i=0
    fusionPred_raw = maxPred_raw.copy()
    for mean, max in zip(meanPred_raw.copy(), maxPred_raw):
        mean_norm = mean / np.max(mean)
        fusionPred_raw[i] = np.multiply(mean_norm, max)
        i+=1

    """for i in range(0,X_test.shape[0]):
        utils_print.printBrief8Cells("title",["img","pred","feat1","feat2", "feat3", "mean","max","fusion"],
                                     [X_test[i],preds[i],pred1_t[i],pred2_t[i],pred3_t[i],meanPred_raw[i],
                                      maxPred_raw[i], fusionPred_raw[i]])"""

    return fusionPred_raw

def augmentedPrediction(model, X, aug_param):
    X_aug = flip(X.copy(), aug_param)
    preds = model.predict(X_aug)
    preds = flip(preds, aug_param)
    return preds

def loadAndPredict(modelName, X_test, modelDir, fold, mode="cla"):
    # Get Trained Model path
    modelPath = selectModel(modelName, modelDir, fold)

    # Loading Trained Model
    print(f'Loading Model | {modelName} | {modelPath}')
    tf.keras.backend.clear_session()
    if mode == "seg":
        custom_objects = {"iou_coef": iou_coef, "dice_coef": dice_coef}
        model = tf.keras.models.load_model(modelPath, compile=False, custom_objects=custom_objects)
    else:
        model = tf.keras.models.load_model(modelPath, compile=False)

    # Predict
    preds = model.predict(X_test)
    tf.keras.backend.clear_session()

    return preds


def dice_coef(y_true, y_pred):
    y_true_f = K.cast(y_true, dtype='float32')  # Converte in float32
    y_pred_f = K.cast(y_pred, dtype='float32')  # Converte in float32
    y_true_f = K.flatten(y_true_f)
    y_pred_f = K.flatten(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred):
    y_true_f = K.cast(y_true, dtype='float32')  # Converte in float32
    y_pred_f = K.cast(y_pred, dtype='float32')  # Converte in float32
    y_true_f = K.flatten(y_true_f)
    y_pred_f = K.flatten(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def iou_coef_loss(y_true, y_pred):
    return 1 - iou_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def gMean(y_true, y_pred):
    smooth = 1e-5
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f) + smooth
    ppv = true_positives / (tf.reduce_sum(y_pred_f) + smooth)
    tpr = true_positives / (tf.reduce_sum(y_true_f) + smooth)
    return tf.math.sqrt(tpr * ppv)


def gMean_loss(y_true, y_pred):
    return 1.0 - gMean(y_true, y_pred)


def mcc(y_true, y_pred):
    """
    Computes the Matthews Correlation Coefficient loss between y_true and y_pred.
    """
    y_true_f = tf.keras.layers.Flatten()(y_true)
    y_pred_f = tf.keras.layers.Flatten()(y_pred)
    # y_pred_f = tf.keras.backend.round(y_pred_f)

    true_positives = tf.reduce_sum(y_true_f * y_pred_f)
    true_negatives = tf.reduce_sum((1 - y_true_f) * (1 - y_pred_f))
    false_positives = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    false_negatives = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    numerator = true_positives * true_negatives - false_positives * false_negatives
    denominator = tf.math.sqrt(
        (true_positives + false_positives) * (true_positives + false_negatives) * (
                true_negatives + false_positives) * (true_negatives + false_negatives))
    mcc = numerator / (denominator + 1e-5)

    return mcc


def mcc_loss(y_true, y_pred):
    return 1.0 - mcc(y_true, y_pred)

