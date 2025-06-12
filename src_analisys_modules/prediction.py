import time

import numpy as np
from classification_models_3D.tfkeras import Classifiers
from numpy import flip
from skimage import morphology
from skimage.transform import resize
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling3D, Dropout, Dense, Activation
from tensorflow.keras.layers import BatchNormalization

import plotter3D
from src.models_builder import build_unet_resnet, build_unet_densenet
from src.uNet.InceptionUnet import get_unet
from src.preprocessing import setGray2Color, thresholdMaskSet, thresholdMask
from src import evaluate, augmentation, voting, plotter
import src.uNet.uNet_Model_V8_v40 as uNets
import tensorflow as tf
import segmentation_models_3D as sm

from src_analisys_modules.datagenerator import array3DToSubCubes
from src_analisys_modules.preprocessing import extract_subcubes, reconstruct_image


def test_time_augmentation_prediction(X_test, mdl):
    X_test_flip0 = flip(X_test.copy(), 0)
    X_test_flip1 = flip(X_test.copy(), 1)
    X_test_flip2 = flip(X_test.copy(), -1)

    ib_pred = mdl.predict(X_test, verbose=False)
    ib_pred_flip0 = mdl.predict(X_test_flip0, verbose=False)
    ib_pred_flip1 = mdl.predict(X_test_flip1, verbose=False)
    ib_pred_flip2 = mdl.predict(X_test_flip2, verbose=False)

    ib_pred_flip0 = flip(ib_pred_flip0, 0)
    ib_pred_flip1 = flip(ib_pred_flip1, 1)
    ib_pred_flip2 = flip(ib_pred_flip2, -1)

    predictions = [ib_pred, ib_pred_flip0, ib_pred_flip1, ib_pred_flip2]

    ib_augVoting_soft, ib_augVoting_soft_raw = \
        voting.segmentation_SoftVoting3(predictions, t=0.5)

    del ib_pred
    del ib_pred_flip0
    del ib_pred_flip1
    del ib_pred_flip2

    return ib_augVoting_soft, ib_augVoting_soft_raw

def segmentation2D_ensemble_prediction(X, model_list, name_list, weights_dir, segm_thr):
    predictions = []
    predictions_aug = []

    # ENSEMBLE TRAINING PHASE ======================================================================================
    # Predict for each model
    print(f"Start Inference...")
    for model, model_name in zip(model_list, name_list):
        # Load Model weights
        model.load_weights(weights_dir + model_name + ".h5")
        print(f"\tCurrent Model: {model_name}")

        # Predict
        preds_raw = model.predict(X, verbose=False)
        _, preds_aug_raw = test_time_augmentation_prediction(X, model)

        # Thresholding
        preds = thresholdMaskSet(preds_raw, t=segm_thr)
        preds_aug = thresholdMaskSet(preds_aug_raw, t=segm_thr)

        # Append to predictions collection to perform voting
        predictions.append(preds)
        predictions_aug.append(preds_aug)

        # VOTING =======================================================================================================
    print("Apply Voting...")

    vot, vot_raw = voting.segmentation_SoftVoting3(predictions, t=segm_thr, samples_axis=0)
    vot_aug, vot_aug_raw = voting.segmentation_SoftVoting3(predictions_aug, t=segm_thr, samples_axis=0)

    return vot_aug

def segmentation3D_ensemble_prediction(X, model_list, name_list, weights_dir, segm_thr):
    predictions = []
    predictions_aug = []

    # ENSEMBLE TRAINING PHASE ======================================================================================
    print(f"Start Inference..:")

    # Predict for each model
    for model, model_name in zip(model_list, name_list):
        print(f"\tCurrent Model: {model_name}")

        # Load Model weights
        model.load_weights(weights_dir + model_name + ".h5")

        # Convert to singleton array
        X_list = []
        X_list.append(X)
        X_list = np.array(X_list)

        # Predict
        preds_raw = model.predict(X_list, verbose=False)
        _, preds_aug_raw = test_time_augmentation_prediction(X_list, model)

        # Thresholding
        preds = thresholdMaskSet(preds_raw, t=segm_thr)
        preds_aug = thresholdMaskSet(preds_aug_raw, t=segm_thr)

        # Append to predictions collection to perform voting
        predictions.append(preds)
        predictions_aug.append(preds_aug)

        # VOTING =======================================================================================================
    print("Apply Voting...")

    vot, vot_raw = voting.segmentation_SoftVoting3(predictions, t=segm_thr, samples_axis=0)
    vot_aug, vot_aug_raw = voting.segmentation_SoftVoting3(predictions_aug, t=segm_thr, samples_axis=0)

    return vot_aug

def segmentation3DSubs_ensemble_prediction(X, model_list, name_list, weights_dir, segm_thr, verbose=False):
    predictions = []
    predictions_aug = []

    # Preprocessing:
    """X_list = []
    X_list.append(X)
    X_list = np.array(X_list)"""
    subcube_size = (64,64,64,1)
    stride = 32

    cube_shape = X.shape
    #print(cube_shape)
    X_decomposed = extract_subcubes(X, subcube_size, stride)
    #print(X_decomposed.shape)
    # ENSEMBLE TRAINING PHASE ======================================================================================
    if verbose: print(f"Start Inference..:")

    # Predict for each model
    for model, model_name in zip(model_list, name_list):
        if verbose: print(f"\tCurrent Model: {model_name}")

        # Load Model weights
        model.load_weights(weights_dir + model_name + ".h5")

        # Convert to singleton array
        # Predict
        preds_raw = model.predict(X_decomposed, verbose=False, batch_size=8)

        #_, preds_aug_raw = test_time_augmentation_prediction(X_decomposed, model)

        # Thresholding
        #preds = thresholdMaskSet(preds_raw, t=segm_thr)
        #preds_aug = thresholdMaskSet(preds_aug_raw, t=segm_thr)

        # Append to predictions collection to perform voting
        pred_reconstruction = reconstruct_image(preds_raw, cube_shape, subcube_size, stride, mode="avg")

        predictions.append(pred_reconstruction)
        #predictions_aug.append(preds_aug_raw)

    # VOTING =======================================================================================================
    if verbose: print("Apply Voting...")

    vot, vot_raw = voting.segmentation_SoftVoting3(predictions, t=segm_thr, samples_axis=0)
    #vot_aug, vot_aug_raw = voting.segmentation_SoftVoting3(predictions_aug, t=segm_thr, samples_axis=0)

    #pred_reconstruction = reconstruct_image(vot_raw, cube_shape, subcube_size, stride, mode="avg")
    #vot_th = thresholdMask(vot_raw, t=segm_thr)

    """predCube0 = pred_reconstruction  # Voting #unet2D
    predCube1 = thresholdMask(pred_reconstruction, segm_thr)  # Voting #unet2D
    plotter3D.plotVolumetricSlices(X, [predCube0, predCube1],
                                   ["Axial", "Coronal", "Sagittal"],
                                   img_mean_projection=False, mask_mean_projection=True,
                                   merge_masks=False)"""

    return vot_raw

def classification3D_ensemble_prediction(X, model_list, name_list, weights_dir, verbose=False):
    predictions = []


    # Start training/test for each model
    if verbose: print(f"Start Inference..:")
    for model, model_name in zip(model_list, name_list):
        if verbose: print(f"\tCurrent Model: {model_name}")

        # Load weights
        model.load_weights(weights_dir+model_name+".h5")

        # Convert to singleton array
        X_list = []
        X_list.append(X)
        X_list = np.array(X_list)

        # Predict
        preds_raw = model.predict(X_list, verbose=False)
        predictions.append(preds_raw)

    # Voting
    if verbose: print("Apply Voting...")
    vot_raw = voting.classification_softVoting(zip(predictions[0], predictions[1], predictions[2]))

    return vot_raw

def detect_lesion_2D(cube, segm_thr=0.5, img_shape=(160,160,1), weights_dir="model_weights/lesion_segmentation/0/"):
    # Preprocessing
    cube = resize(cube, (96,160,160,1), mode='constant')

    # LOAD MODELS ==================================================================================================
    print("Loading Models...")
    # Build Models:
    print("Building CustomUNet...")
    custom_model, custom_name = uNets.build_model(*img_shape, 1), "custom"
    tf.keras.backend.clear_session()

    print("Building Inception-UNet...")
    inception_model, inception_name = get_unet(*img_shape), "inception"
    tf.keras.backend.clear_session()

    print("Building Densenet121-UNet...")
    densenet_model, densenet_name = build_unet_densenet(img_shape, out_channels=1,weights=None), "densenet121"
    tf.keras.backend.clear_session()

    # Combine selected models
    models = [custom_model, densenet_model, inception_model]
    model_names = [custom_name, densenet_name, inception_name]

    ensemble_pred = segmentation2D_ensemble_prediction(cube, models, model_names, weights_dir, segm_thr)
    return ensemble_pred


def detect_lesion_3D(cube, segm_thr=0.5, input_shape=(96, 96, 96, 1), weights_dir="model_weights/lesion_segmentation/1/"):
    # Preprocessing
    cube = resize(cube, input_shape, mode='constant')

    print("Loading Models...")
    model_1, model_name_1 = sm.Unet('resnet50', encoder_weights=None, input_shape=input_shape, activation='sigmoid'), \
                            "Unet-resnet50"
    print(f"\tModel built: {model_name_1}")

    model_3, model_name_3 = sm.Linknet('densenet121', encoder_weights=None, input_shape=input_shape,
                                       classes=1, activation='sigmoid'), "Linknet-densenet121"
    print(f"\tModel built: {model_name_3}")

    model_4, model_name_4 = sm.Unet('densenet121', encoder_weights=None, input_shape=input_shape), "Unet-densenet121"
    print(f"\tModel built: {model_name_4}")

    model_6, model_name_6 = sm.Unet('inceptionv3', encoder_weights=None, input_shape=input_shape), "Unet-inceptionv3"
    print(f"\tModel built: {model_name_6}")

    models = [model_1, model_3, model_4, model_6]
    model_names = [model_name_1, model_name_3, model_name_4, model_name_6]

    ensemble_pred = segmentation3D_ensemble_prediction(cube, models, model_names, weights_dir, segm_thr)

    return ensemble_pred[0]

def detect_lesion_3D_subcubes(cube, Log, segm_thr=0.5,cube_shape=(128, 128, 128, 1), input_shape=(64, 64, 64, 1),
                                 weights_dir="model_weights/lesion_segmentation/3DSub_0/", verbose=False):



    if verbose: print("AI MODULE - Neoplasia Segmentation -")
    start = time.time()

    # Preprocessing
    if verbose: print("Apply Preprocessing...")
    Log.write(None, None, "Preprocessing del DICOM", "1/3")
    cube = resize(cube, cube_shape, mode='constant')

    # LOAD MODELS ==================================================================================================
    if verbose: print("Loading Models...")
    Log.write(None, None, "Caricamento dei modelli IA", "2/3")

    ew = None
    model_1, model_name_1 = sm.Unet('resnet50', encoder_weights=ew, input_shape=input_shape), "Unet-resnet50"
    if verbose: print(f"\tModel built: {model_name_1}")

    model_2, model_name_2 = sm.Linknet('densenet121', encoder_weights=ew, input_shape=input_shape), "Linknet-densenet121"
    if verbose: print(f"\tModel built: {model_name_2}")

    model_3, model_name_3 = sm.Unet('densenet121', encoder_weights=ew, input_shape=input_shape), "Unet-densenet121"
    if verbose: print(f"\tModel built: {model_name_3}")

    model_4, model_name_4 = sm.Unet('inceptionv3', encoder_weights=ew, input_shape=input_shape),  "Unet-inceptionv3"
    if verbose: print(f"\tModel built: {model_name_4}")

    models = [model_1,model_2,model_3,model_4]
    model_names = [model_name_1,model_name_2,model_name_3,model_name_4]

    Log.write(None, None, "Segmentazione della neoplasia (potrebbe richiedere qualche minuto)", "3/3")
    ensemble_pred_raw = segmentation3DSubs_ensemble_prediction(cube, models, model_names, weights_dir, segm_thr,
                                                           verbose=verbose)
    ensemble_pred = thresholdMask(ensemble_pred_raw, t=segm_thr)

    prob = np.sum(ensemble_pred_raw*ensemble_pred)/np.count_nonzero(ensemble_pred)
    if verbose: print(f"Neoplasia Segmentation Confidence : {prob} %")

    if verbose: print(f"Elapsed Time - Neoplasia Segmentation : {time.time() - start} seconds")

    return ensemble_pred


def detect_lymphnode_2D(cube, segm_thr=0.5, input_shape=(160, 160, 1),
                        weights_dir="model_weights/lymphnode_segmentation/0/", verbose=False):

    if verbose: print("AI MODULE - Lymph Nodes Segmentation -")
    start = time.time()

    # Preprocessing
    if verbose: print("Apply Preprocessing...")
    cube = resize(cube, (96, *input_shape), mode='constant')

    # LOAD MODELS ==================================================================================================
    if verbose: print("Loading Models...")

    # Build Models:
    custom_model, custom_name = uNets.build_model(*input_shape, 1), "custom"
    if verbose: print(f"\tModel built: {custom_name}")
    tf.keras.backend.clear_session()

    inception_model, inception_name = get_unet(*input_shape), "inception"
    if verbose: print(f"\tModel built: {inception_name}")
    tf.keras.backend.clear_session()

    densenet_model, densenet_name = build_unet_densenet(input_shape, out_channels=1, weights=None), "densenet121"
    if verbose: print(f"\tModel built: {densenet_name}")
    tf.keras.backend.clear_session()

    # Combine selected models
    models = [custom_model, densenet_model, inception_model]
    model_names = [custom_name, densenet_name, inception_name]

    ensemble_pred = segmentation2D_ensemble_prediction(cube, models, model_names, weights_dir, segm_thr)
    return ensemble_pred

def detect_lymphnode_3D_subcubes(cube, Log, segm_thr=0.5,cube_shape=(160, 160, 160, 1), input_shape=(64, 64, 64, 1),
                                 weights_dir="model_weights/lymphnode_segmentation/3DSub_0/", verbose=False):

    if verbose: print("AI MODULE - Lymph Nodes Segmentation -")
    start = time.time()

    # Preprocessing
    if verbose: print("Apply Preprocessing...")
    Log.write(None, None, "Preprocessing del DICOM", "1/3")
    cube = resize(cube, cube_shape, mode='constant')

    # LOAD MODELS ==================================================================================================
    if verbose: print("Loading Models...")
    Log.write(None, None, "Caricamento dei modelli IA", "2/3")
    ew = None
    model_1, model_name_1 = sm.Unet('resnet50', encoder_weights=ew, input_shape=input_shape), "Unet-resnet50"
    if verbose: print(f"\tModel built: {model_name_1}")

    model_2, model_name_2 = sm.Linknet('densenet121', encoder_weights=ew, input_shape=input_shape), "Linknet-densenet121"
    if verbose: print(f"\tModel built: {model_name_2}")

    model_3, model_name_3 = sm.Unet('densenet121', encoder_weights=ew, input_shape=input_shape), "Unet-densenet121"
    if verbose: print(f"\tModel built: {model_name_3}")

    model_4, model_name_4 = sm.Unet('inceptionv3', encoder_weights=ew, input_shape=input_shape), "Unet-inceptionv3"
    if verbose: print(f"\tModel built: {model_name_4}")

    models = [model_1,model_2,model_3,model_4]
    model_names = [model_name_1,model_name_2,model_name_3,model_name_4]

    Log.write(None, None, "Segmentazione dei linfonodi (potrebbe richiedere qualche minuto)", "3/3")
    ensemble_pred_raw = segmentation3DSubs_ensemble_prediction(cube, models, model_names, weights_dir, segm_thr,
                                                           verbose=verbose)
    ensemble_pred_t02 = thresholdMask(ensemble_pred_raw, t=0.2)
    ensemble_pred_t05 = thresholdMask(ensemble_pred_raw, t=0.5)
    ensemble_pred = morphology.reconstruction(ensemble_pred_t05, ensemble_pred_t02)

    prob = np.sum(ensemble_pred_raw*ensemble_pred)/np.count_nonzero(ensemble_pred)
    if verbose: print(f"Lymph Nodes Segmentation Confidence : {prob} %")

    if verbose: print(f"Elapsed Time - Lymph Nodes Segmentation: {time.time() - start} seconds")

    return ensemble_pred


def estimate_infiltration_stage(cube, Log, input_shape=(96, 160, 160, 1), verbose=False,
                                weights_dir="model_weights/tstage_classification/0/"):
    if verbose: print("AI MODULE - Stage T Classification -")
    start = time.time()

    # Preprocessing
    if verbose: print("Apply Preprocessing...")
    Log.write(None, None, "Preprocessing del DICOM", "1/3")
    cube = resize(cube, input_shape, mode='constant')

    # Build Models:
    def editClassifier(model):
        x = model.layers[-1].output
        x = GlobalAveragePooling3D()(x)
        x = Dropout(0.3)(x)
        x = Dense(5)(x)
        x = Activation('softmax')(x)
        return Model(inputs=model.inputs, outputs=x)

    # Load Models
    if verbose: print("Loading Models...")
    Log.write(None, None, "Caricamento dei modelli IA", "2/3")

    ResNet18, preprocess_input = Classifiers.get('resnet18')
    model1, model_name1 = ResNet18(input_shape=input_shape, weights=None, ), "resnet18"
    model1 = editClassifier(model1)
    if verbose: print(f"\tModel built: {model_name1}")
    tf.keras.backend.clear_session()

    Densenet121, _ = Classifiers.get('resnet18')
    model2, model_name2 = Densenet121(input_shape=input_shape, weights=None), "densenet121"
    model2 = editClassifier(model2)
    if verbose: print(f"\tModel built: {model_name2}")
    tf.keras.backend.clear_session()

    InceptionV3, _ = Classifiers.get('inceptionv3')
    model3, model_name3 = InceptionV3(input_shape=input_shape, weights=None), "inceptionv3"
    model3 = editClassifier(model3)
    if verbose: print(f"\tModel built: {model_name3}")
    tf.keras.backend.clear_session()

    models = [model1, model2, model3]
    model_names = [model_name1, model_name2, model_name3]

    Log.write(None, None, "Classificazione del grado di infiltrazione (potrebbe richiedere qualche minuto)", "3/3")
    ensemble_pred = classification3D_ensemble_prediction(cube, models, model_names, weights_dir)
    t_stage = ensemble_pred[0].tolist().index(max(ensemble_pred[0]))
    t_stage_confidence = round(max(ensemble_pred[0]),2)

    if verbose: print(f"Predicted T-Stage: {t_stage}")
    if verbose: print(f"T-Stage Classification Confidence : {t_stage_confidence} %")
    if verbose: print(f"Elapsed Time - T Stage Classification: {time.time() - start} seconds")

    return t_stage, t_stage_confidence

def estimate_infiltration_stage_v2(cube, Log, input_shape=(160, 160, 160, 1), verbose=False,
                                weights_dir="model_weights/tstage_classification/1/"):
    if verbose: print("AI MODULE - Stage T Classification -")
    start = time.time()

    # Preprocessing
    if verbose: print("Apply Preprocessing...")
    Log.write(None, None, "Preprocessing del DICOM", "1/3")
    cube = resize(cube, input_shape, mode='constant')

    # Build Models:
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

    # Load Models
    if verbose: print("Loading Models...")
    Log.write(None, None, "Caricamento dei modelli IA", "2/3")

    ResNet50, preprocess_input = Classifiers.get('resnet50')
    model1, model_name1 = ResNet50(input_shape=input_shape, weights=None, ), "resnet50"
    model1 = editClassifier(model1)
    if verbose: print(f"\tModel built: {model_name1}")
    tf.keras.backend.clear_session()

    Densenet121, _ = Classifiers.get('densenet121')
    model2, model_name2 = Densenet121(input_shape=input_shape, weights=None), "densenet121"
    model2 = editClassifier(model2)
    if verbose: print(f"\tModel built: {model_name2}")
    tf.keras.backend.clear_session()

    InceptionV3, _ = Classifiers.get('inceptionv3')
    model3, model_name3 = InceptionV3(input_shape=input_shape, weights=None), "inceptionv3"
    model3 = editClassifier(model3)
    if verbose: print(f"\tModel built: {model_name3}")
    tf.keras.backend.clear_session()

    ResNet50, preprocess_input = Classifiers.get('resnext50')
    model4, model_name4 = ResNet50(input_shape=input_shape, weights=None, ), "resnext50"
    model4 = editClassifier(model4)
    if verbose: print(f"\tModel built: {model_name4}")
    tf.keras.backend.clear_session()

    Densenet121, _ = Classifiers.get('seresnet50')
    model5, model_name5 = Densenet121(input_shape=input_shape, weights=None), "seresnet50"
    model5 = editClassifier(model5)
    if verbose: print(f"\tModel built: {model_name5}")
    tf.keras.backend.clear_session()

    InceptionV3, _ = Classifiers.get('inceptionresnetv2')
    model6, model_name6 = InceptionV3(input_shape=input_shape, weights=None), "inceptionresnetv2"
    model6 = editClassifier(model6)
    if verbose: print(f"\tModel built: {model_name6}")
    tf.keras.backend.clear_session()

    models = [model1, model2, model3, model4, model5, model6]
    model_names = [model_name1, model_name2, model_name3, model_name4, model_name5, model_name6]

    Log.write(None, None, "Classificazione del grado di infiltrazione (potrebbe richiedere qualche minuto)", "3/3")
    ensemble_pred = classification3D_ensemble_prediction(cube, models, model_names, weights_dir)
    t_stage = ensemble_pred[0].tolist().index(max(ensemble_pred[0]))
    t_stage_confidence = round(max(ensemble_pred[0]),2)

    if verbose: print(f"Predicted T-Stage: {t_stage}")
    if verbose: print(f"T-Stage Classification Confidence : {t_stage_confidence} %")
    if verbose: print(f"Elapsed Time - T Stage Classification: {time.time() - start} seconds")

    return t_stage, t_stage_confidence



