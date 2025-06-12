from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling3D, Dropout, Dense, Activation
from tensorflow.keras.layers import BatchNormalization

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

def editClassifier(model):
    x = model.layers[-1].output
    x = GlobalAveragePooling3D()(x)

    x = Dense(512, activation=None, kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation(activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(5)(x)
    x = Activation('softmax')(x)
    return Model(inputs=model.inputs, outputs=x)