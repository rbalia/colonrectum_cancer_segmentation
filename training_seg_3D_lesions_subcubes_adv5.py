import gc
import os
import sys
import numpy as np
from skimage import morphology
from skimage.transform import resize
import pandas as pd
from src.evaluate import customModelScore, evaluateGeometricMean
from src.models_actions import iou_coef, dice_coef
from src.models_builder import dice_loss
from src.splitter import getFoldIdx
from src.preprocessing import thresholdMaskSet
from src import evaluate, voting
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow.keras.backend as K
import segmentation_models_3D as sm

# Manage script inputs
from src_analisys_modules.datagenerator import DataGeneratorAOU_fromList, datagenTo3DArray, \
    array3DToSubCubes, DataGeneratorAOU_fixedSize
from src_analisys_modules.preprocessing import filter_empty_cubes, reconstruct_dataset_from_subs

print(f"Execution Script '{os.path.basename(__file__)}' | Script Setting:")
print(tf.__version__)
print(tf.config.list_physical_devices(device_type=None))

if len(sys.argv) > 1 and sys.argv[1] in ["train", "test"]:
    script_mode = sys.argv[1]
    print(f"Script Mode: {script_mode}")
else:
    print("Not valid argument for 'script_mode'")
    script_mode = "test"
    print(f"Setting Default Mode: {script_mode} ")
    # exit(0)

if len(sys.argv) > 2 and sys.argv[2] in ["0", "1"]:
    target_gpu = sys.argv[2]
    print(f"Selected GPU: {target_gpu}")
else:
    print("Not valid argument for 'target_gpu'")
    target_gpu = "0"
    print(f"Setting Default GPU: {target_gpu}")

if len(sys.argv) > 3:
    experiment_name = sys.argv[3]
    print(f"Experiment name: {experiment_name}")
else:
    print("Not valid argument for 'experiment_name'")
    experiment_name = "default"
    print(f"Setting Default experiment name: {experiment_name}")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = f"{target_gpu}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[int(target_gpu)], 'GPU')
#logical_devices = tf.config.list_logical_devices('GPU')


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 3000)

K.set_image_data_format('channels_last')

if __name__ == '__main__':

    # Define Script Parameters =========================================================================================
    im_size = 64
    n_slice = 160
    cube_orig_shape = (160,160,160)
    cube_resz_shape = (128,128,128,1)
    subcube_shape = (64,64,64,1)
    stride = 32
    segm_thr = 0.5
    detc_thr = 0.5
    script_name = os.path.basename(__file__).split(".")[0]
    evaluations_global = {'Fold': [], 'Method': [], 'Threshold':[], 'Protocol': [], 'Setting': [],
                          'IOU': [], 'Dice': [], 'TNR':[], 'TPR':[], 'FDR':[],
                          'd-Acc': [], 'd-BAcc': [], 'd-TPR': [], 'd-TNR': [],
                          'MyScore':[], 'G-Mean':[], 'MCC':[]}

    print("Script Parameters Brief:")
    print(f"\tScript name:            {script_name}")
    print(f"\tExperiment name:        {experiment_name}")
    print(f"\tSegmentation threshold: {segm_thr}")
    print(f"\tDetection threshold:    {detc_thr}")

    # Load Dataset =====================================================================================================
    print("Loading Dataset..")
    csvDataset = pd.read_csv(r'dataset_colonrectum/binaryPkg/dataset_paths.csv')
    csvDataset_CA = pd.read_csv(r'dataset_colonrectum/binaryPkg_CA/dataset_paths.csv')
    X_paths = []
    Y_paths = []

    for i, row in csvDataset.iterrows():
        # if os.path.isfile(str(row["lymphnode_nrrd"])):
        if len(str(row["neoplasia_nrrd"])) > 10:
            print(row["neoplasia_nrrd"])
            X_paths.append(row["image_npy"])
            Y_paths.append(row["neoplasia_npy"])
    n_images = len(X_paths)

    for i, row in csvDataset_CA.iterrows():
        # if os.path.isfile(str(row["lymphnode_nrrd"])):
        if len(str(row["neoplasia_nrrd"])) > 10:
            print(row["neoplasia_nrrd"])
            X_paths.append(row["image_npy"])
            Y_paths.append(row["neoplasia_npy"])
    n_images_CA = len(X_paths) - n_images

    print(f"\tNumber of images: {n_images + n_images_CA}")

    # Dataset 5-Fold Split and filtering ===============================================================================
    n_folds = 5
    iteration = 0
    crossval_iterations = 5

    params = {'full_cube_shape': cube_resz_shape, 'subs_cube_shape': subcube_shape[0:-1], 'stride': stride,
              'batch_size': 8, 'n_channels': 1, 'n_classes': 1, 'n_cubes': 4, 'shuffle': True, 'n_aug': 1,
              'apply_augmentation':False}

    #params = {'full_cube_shape': cube_resz_shape, 'subs_cube_shape' : subcube_shape[0:-1], 'stride':stride,
    #          'batch_size' : 8, 'n_channels': 1, 'n_classes': 1, 'n_cubes': 4, 'shuffle': True, 'n_aug': 10,
    #          'apply_augmentation': True,
    #          'extraction_setting':[
    #              {'cube_shape':(128,128,128,1), 'stride':32},
    #              #{'cube_shape':(96,96,96,1), 'stride':32}
    #          ]}

    params_test_adv = {'full_cube_shape': cube_resz_shape, 'subs_cube_shape': subcube_shape[0:-1], 'stride': stride,
              'batch_size': 8, 'n_channels': 1, 'n_classes': 1, 'n_cubes': 4, 'shuffle': False, 'n_aug': 1,
                       'apply_augmentation':False}

    params_test = {'dim': cube_orig_shape, 'batch_size': 1, 'n_classes': 1, 'n_channels': 1, 'shuffle': False}

    foldIdx = getFoldIdx(n_folds, range(0, n_images))
    foldIdx_CA = getFoldIdx(n_folds, range(0, n_images_CA))

    # For each fold ...
    while iteration < crossval_iterations:
        curr_fold = iteration % n_folds

        # Split dataset
        #datagen_test = DataGeneratorAOU(foldIdx[curr_fold][1], **params_test)
        datagen_validation = DataGeneratorAOU_fromList(
            #foldIdx[curr_fold][1], X_paths, Y_paths, **params_test)
            np.concatenate((foldIdx[curr_fold][1], foldIdx_CA[curr_fold][1] + n_images)), X_paths, Y_paths, **params_test)
        #datagen_validation2 = DataGeneratorAOU_fromList_adv(foldIdx[curr_fold][1], X_paths, Y_paths, **params_test_adv)
        if script_mode == "train":
            #valset_splitpoint = int((len(foldIdx[curr_fold][0])) / 100 * 80)
            #datagen_training = DataGeneratorAOU(foldIdx[curr_fold][0][0:valset_splitpoint], **params)
            #datagen_validation = DataGeneratorAOU(foldIdx[curr_fold][0][valset_splitpoint:], **params)
            datagen_training = DataGeneratorAOU_fixedSize(foldIdx[curr_fold][0], X_paths, Y_paths, **params)

        # CONVERT TO 2D ================================================================================================
        X_val, Y_val = datagenTo3DArray(datagen_validation)
        #X_val_144 = resize(X_val, ((X_val.shape[0],) + (144,144,144, 1)), mode='constant')
        #Y_val_144 = resize(Y_val, ((Y_val.shape[0],) + (144,144,144, 1)), order=0, anti_aliasing=False)
        X_val_128 = resize(X_val, ((X_val.shape[0],) + (128, 128, 128, 1)), mode='constant')
        Y_val_128 = resize(Y_val, ((Y_val.shape[0],) + (128, 128, 128, 1)), order=0, anti_aliasing=False)
        #X_val_112 = resize(X_val, ((X_val.shape[0],) + (112,112,112, 1)), mode='constant')
        #Y_val_112 = resize(Y_val, ((Y_val.shape[0],) + (112,112,112, 1)), order=0, anti_aliasing=False)
        X_val_96 =  resize(X_val, ((X_val.shape[0],) + (96, 96, 96, 1)), mode='constant')
        Y_val_96 =  resize(Y_val, ((Y_val.shape[0],) + (96, 96, 96, 1)), order=0, anti_aliasing=False)
        #X_val_80 =  resize(X_val, ((X_val.shape[0],) + (80,80,80, 1)), mode='constant')
        #Y_val_80 =  resize(Y_val, ((Y_val.shape[0],) + (80,80,80, 1)), order=0, anti_aliasing=False)

        #X_val_nofiltr_144, Y_val_nofiltr_144 = array3DToSubCubes(X_val_144, Y_val_144, subcube_shape, stride)
        X_val_nofiltr_128, Y_val_nofiltr_128 = array3DToSubCubes(X_val_128, Y_val_128, subcube_shape, stride)
        #X_val_nofiltr_112, Y_val_nofiltr_112 = array3DToSubCubes(X_val_112, Y_val_112, subcube_shape, stride)
        X_val_nofiltr_96, Y_val_nofiltr_96 = array3DToSubCubes(X_val_96, Y_val_96, subcube_shape, stride)
        #X_val_nofiltr_80, Y_val_nofiltr_80 = array3DToSubCubes(X_val_80, Y_val_80, subcube_shape, stride)
        X_val_nofiltr = np.concatenate((X_val_nofiltr_128, X_val_nofiltr_96))
        Y_val_nofiltr = np.concatenate((Y_val_nofiltr_128, Y_val_nofiltr_96))
        X_val,Y_val = filter_empty_cubes(X_val_nofiltr,Y_val_nofiltr)

        print(f"Number of Validation Samples:      {len(X_val)}   \tX_Shape:{X_val.shape}   \tY_Shape:{Y_val.shape}")

        # DEBUG DATA GENERATOR =========================================================================================
        """print("Loading Training Data")
        for batch_id, batch in enumerate(datagen_training):
            print(f"Batch: {batch_id} - X size:{len(batch[0])} - Y size:{len(batch[1])}")
            for x, y in zip(batch[0], batch[1]):
                plotter3D.plotVolumetricSlices(x,[y],axis_name=["Ax", "Cor", "Sag"], mask_mean_projection=True)
                break

        exit(0)"""
        # LOAD MODELS ==================================================================================================
        def load_models(id, in_shape):
            print("Loading Models...")
            ew = None
            if id == 0:
                model_1, model_name_1 = sm.Unet('resnet50', encoder_weights=ew, input_shape=in_shape,
                                                activation='sigmoid'), \
                    "Unet-resnet50"
                print(f"\tModel built: {model_name_1}")
                return model_1, model_name_1

            if id == 1:
                model_2, model_name_2 = sm.Linknet('densenet121', encoder_weights=ew, input_shape=in_shape,
                                                   classes=1, activation='sigmoid'), \
                    "Linknet-densenet121"
                print(f"\tModel built: {model_name_2}")
                return model_2, model_name_2

            if id == 2:
                model_3, model_name_3 = sm.Unet('densenet121', encoder_weights=ew, input_shape=in_shape), \
                    "Unet-densenet121"
                print(f"\tModel built: {model_name_3}")
                return model_3, model_name_3

            # If you need to specify non-standard input shape
            if id == 3:
                model_4, model_name_4 = sm.Unet('inceptionv3', encoder_weights=ew, input_shape=in_shape), \
                    "Unet-inceptionv3"
                print(f"\tModel built: {model_name_4}")
                return model_4, model_name_4


        models_to_load = [0,1,2,3]
        loaded_model_names = []

        predictions_144 = []
        predictions_128 = []
        predictions_112 = []
        predictions_96 = []
        predictions_80 = []
        predictions_Join = []
        predictions_aug = []

        # ENSEMBLE TRAINING PHASE ======================================================================================
        # Start training/test for each model
        for model_id in models_to_load:
            model, model_name = load_models(model_id, subcube_shape)
            loaded_model_names.append(model_name)
            tf.keras.backend.clear_session()
            gc.collect()
            #if model_name == "Unet-inceptionv3":
            #script_name = "training_seg_3D_lesions_subcubes_adv3"
            #script_name = "training_seg_3D_lesions_subcubes_adv4_dualSize"
            checkpoint_path = f"models/{script_name}/{experiment_name}/{model_name}_fold{curr_fold}.h5"
            #checkpoint_path = f"models/training_seg_3D_lymphnode_resize3/{experiment_name}/{model_name}_fold{curr_fold}.h5"

            if script_mode == "train":
                print(f"Fold {curr_fold} - Start Training - Current Model: {model_name}")

                # Set training parameters and compile
                loss = dice_loss#"binary_crossentropy"#"msle" #"hinge"
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)  # "adam"
                metrics = [iou_coef, dice_coef]
                model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

                # Define callbacks
                es = EarlyStopping(monitor='val_dice_coef', mode='max', verbose=1, patience=50)
                mc = ModelCheckpoint(checkpoint_path, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1)
                callbacks = [es, mc]

                # Train
                model_history = model.fit(datagen_training, verbose=True, epochs=150,
                                          validation_data=(X_val, Y_val), shuffle=False, callbacks=callbacks)

                """plotter.plotSegmentationHistory(model_history, model_name, mode="save",
                           save_path=f"models/{script_name}/{experiment_name}/{model_name}_fold{curr_fold}_history.png")"""
                #tf.keras.backend.clear_session()

            model.load_weights(checkpoint_path)

            # PREDICTION ===============================================================================================
            print(f"Start Inference - Current Model: {model_name}")
            #X_test_144, Y_test_144 = X_val_nofiltr_144, Y_val_nofiltr_144
            X_test_128, Y_test_128 = X_val_nofiltr_128, Y_val_nofiltr_128
            #X_test_112, Y_test_112 = X_val_nofiltr_112, Y_val_nofiltr_112
            #X_test_96, Y_test_96 = X_val_nofiltr_96, Y_val_nofiltr_96
            #X_test_80, Y_test_80 = X_val_nofiltr_80, Y_val_nofiltr_80

            #tf.keras.backend.clear_session()
            #gc.collect()
            def predict_on_batch(model, x, cube_size, verbose=False, batch_size=8, ):
                batch_size = 8
                preds = []
                #print(f"Prediction on cube sized {cube_size}px")
                for i in range(0, len(x), batch_size):
                    #print(f"Prediction on batch #{i} for cube sized at {cube_size}")
                    batch_x = x[i:i + batch_size]
                    batch_preds = model.predict(batch_x, verbose=False, batch_size=batch_size)
                    preds.append(batch_preds)
                #print("Concatenazione")
                return np.concatenate(preds, axis=0)
            #model.load_weights(checkpoint_path)
            #preds_raw_144 = model.predict(X_test_144, verbose=False, batch_size=8)
            preds_raw_128 = predict_on_batch(model, X_test_128,  "128", verbose=False, batch_size=8)#model.predict(X_test_128, verbose=False, batch_size=8)
            #preds_raw_112 = model.predict(X_test_112, verbose=False, batch_size=8)
            #preds_raw_96 = model.predict(X_test_96, verbose=False, batch_size=8)
            #preds_raw_80 = model.predict(X_test_80, verbose=False, batch_size=8)

            # Append to predictions collection to perform voting
            #predictions_144.append(preds_raw_144)
            predictions_128.append(preds_raw_128)
            #predictions_112.append(preds_raw_112)
            #predictions_96.append(preds_raw_96)
            #predictions_80.append(preds_raw_80)

            tf.keras.backend.clear_session()
            gc.collect()

        # ORIGINAL CUBE RECONSTRUCTION =================================================================================
        X_test_rec_128, Y_test_rec_128, predictions_rec_128 = \
            reconstruct_dataset_from_subs(X_test_128, Y_test_128, predictions_128, (128, 128, 128, 1),
                                          subcube_shape, stride)

        predictions_rec_128 = predictions_rec_128.tolist()

        # VOTING =======================================================================================================
        print("Apply Voting...")

        vot_128, vot_raw_128 = voting.segmentation_SoftVoting3(predictions_rec_128, t=segm_thr, samples_axis=0)

        vot_raw_th2 = thresholdMaskSet(vot_raw_128, t=0.2)
        vot_raw_th5 = thresholdMaskSet(vot_raw_128, t=0.5)

        # esecuzione della ricostruzione per erosione
        vot_mean_rec = morphology.reconstruction(vot_raw_th5, vot_raw_th2)

        predictions_rec_128.append(vot_raw_th2)
        predictions_rec_128.append(vot_raw_th5)
        predictions_rec_128.append(vot_mean_rec)

        """for img, gt, vot in zip(X_test_rec_128, Y_test_rec_128, vot_128):
            tpr = evaluate.evaluateTruePositiveRate(vot, gt)
            fdr = evaluate.evaluateFalseDiscoveryRate(vot, gt)
            iou = evaluate.evaluateIOU(vot, gt)
            dice = evaluate.evaluateDice(vot, gt)
            print(f"tpr: {tpr}")
            print(f"fdr: {fdr}")
            print(f"iou: {iou}")
            print(f"dic: {dice}")

            #fmi1 = evaluate.evaluateFowlkesMallowsIndex(vot, gt)
            #fmi2 = evaluate.evaluateFowlkesMallowsIndex2(vot, gt)
            #fmi3 = evaluate.evaluateFowlkesMallowsIndex3(vot, gt)
            #fmi4 = fowlkes_mallows_score(gt.flatten(), vot.flatten())
            #print(f"fmi1: {fmi1}")
            #print(f"fmi2: {fmi2}")
            #print(f"fmi3: {fmi3}")
            #print(f"fmi4: {fmi4}")
            plotter3D.plotVolumetricSlices(img, [gt, vot],
                                           ["Axial", "Coronal", "Sagittal"],
                                           img_mean_projection=False, mask_mean_projection=True,
                                           merge_masks=False, figure_title="Single Net (128)")

            eval_full_128 = evaluate.evaluateSegmentation([gt], [vot],
                                                          t=segm_thr, det_t=segm_thr)
            dif = gt - vot
            gt_1=np.rot90(gt, k=3, axes=(1, 2))
            vot_1=np.rot90(vot, k=3, axes=(1, 2))
            dif_1 = np.rot90(dif, k=3, axes=(1, 2))
            plot_3d_interactive(gt_1, vot_1, eval_full_128["dice"][0], segm_thr)
            x = input()"""


        # EVALUATION ===================================================================================================
        print(f"Fold {curr_fold} - Evaluating Predictions...")
        evaluations = []
        evaluations_currFold = {'Fold': [], 'Method': [], 'Threshold':[], 'Protocol': [], 'Setting': [],
                          'IOU': [], 'Dice': [], 'TNR':[], 'TPR':[], 'FDR':[],
                          'd-Acc': [], 'd-BAcc': [], 'd-TPR': [], 'd-TNR': [],
                          'MyScore':[], 'G-Mean':[], 'MCC':[]}
        #dict.fromkeys(evaluations_global.keys(), [])

        print("\tEvaluating 2D & 3D")
        # Evaluate Predictions and Voting

        for preds_rec_128, mdl_name in zip(predictions_rec_128, loaded_model_names +
                                                                ["Vot_Mean02","Vot_Mean05","Vot_MeanMorph"]):

            eval_full_128 = evaluate.evaluateSegmentation(Y_test_rec_128, np.copy(preds_rec_128),
                                                          t=segm_thr, det_t=segm_thr)

            """for pID, pred in enumerate(preds_rec_128):
                plot_3d_interactive(preds_rec_128[pID], Y_test_rec_128[pID], eval_full_128["dice"][pID], segm_thr)
                x = input()"""
            # Store evaluations
            evaluations.append(("3D-FullCube-Based", "(128)", mdl_name, segm_thr, eval_full_128))

            # Reformat prediction and compute means
        for protocol_name, setting_name, model_name, thr, eval in evaluations:
            # Evaluation identifier by: Fold / Model / Evaluation Protocol / Data Setting
            evaluations_currFold["Fold"].append(curr_fold)
            evaluations_currFold["Method"].append(model_name)
            evaluations_currFold["Threshold"].append(thr)
            evaluations_currFold["Protocol"].append(protocol_name)
            evaluations_currFold["Setting"].append(setting_name)
            # Segmentation Metrics
            evaluations_currFold["IOU"].append(round(np.mean(eval["iou"]) * 100, 2)),
            evaluations_currFold["Dice"].append(round(np.mean(eval["dice"]) * 100, 2)),
            evaluations_currFold["TNR"].append(round(np.mean(eval["tnr"]) * 100, 2)),
            evaluations_currFold["TPR"].append(round(np.mean(eval["tpr"]) * 100, 2)),
            evaluations_currFold["FDR"].append(round(np.mean(eval["fdr"]) * 100, 2)),
            # Detection Metrics
            evaluations_currFold["d-Acc"].append(round(np.mean(eval["det-acc"]) * 100, 2)),
            evaluations_currFold["d-BAcc"].append(round(np.mean(eval["det-bacc"]) * 100, 2)),
            evaluations_currFold["d-TPR"].append(round(np.mean(eval["det-tpr"]) * 100, 2)),
            evaluations_currFold["d-TNR"].append(round(np.mean(eval["det-tnr"]) * 100, 2)),
            # Global Score
            score = customModelScore(np.mean(eval["iou"]), np.mean(eval["dice"]),
                                     np.mean(eval["tpr"]), 1 - np.mean(eval["fdr"]))
            gmean = evaluateGeometricMean(np.mean(eval["tpr"]), 1 - np.mean(eval["fdr"]))
            evaluations_currFold["MyScore"].append(round(score * 100, 2))
            evaluations_currFold["G-Mean"].append(round(gmean * 100, 2))
            evaluations_currFold["MCC"].append(round(np.mean(eval["mcc"]) * 100, 2))

            print(f'Aug/Patch,{model_name},{curr_fold},"{eval["dice"]}"')


        # PLOT RESULT ==================================================================================================
        # Volumetric Visualization

        """for i in range(0,len(X_test_rec_128)):
            imgCube = X_test_rec_128[i]
            gtCube =  Y_test_rec_128[i]

            predCube0 = predictions_rec_128[0][i] # custom # custom874
            predCube1 = predictions_rec_128[1][i] # unet2D # custom
            predCube2 = predictions_rec_128[2][i]  # custom # custom874
            predCube3 = predictions_rec_128[3][i]  # unet2D # custom
            plotter3D.plotVolumetricSlices(imgCube, [gtCube, predCube0, predCube1, predCube2, predCube3],
                                           ["Axial", "Coronal", "Sagittal"],
                                           img_mean_projection=False, mask_mean_projection=True,
                                           merge_masks=False, figure_title="Single Nets")

            vot_t02_id = 4
            vot_t05_id = 5
            vot_rec_id = 6

            predCube0 = predictions_rec_128[vot_t02_id][i] #Voting #unet2D
            predCube1 = predictions_rec_128[vot_t05_id][i]  # Voting #unet2D
            predCube2 = predictions_rec_128[vot_rec_id][i]  # Voting #unet2D
            VotRaw = vot_raw_128[i]
            plotter3D.plotVolumetricSlices(imgCube, [gtCube, VotRaw, predCube0,predCube1,predCube2],
                                           ["Axial", "Coronal", "Sagittal"],
                                           img_mean_projection=False, mask_mean_projection=True,
                                           merge_masks=False)
            
            # Interactive Plotting
            overlay_im = imgCube[..., 0] * 0.3
            overlay_im = np.expand_dims(overlay_im, axis=-1)
            overlay_im = np.repeat(overlay_im, 3, axis=-1)
            overlay_im[..., [0]] = np.maximum(overlay_im[..., [0]], thresholdMask(VotRaw, segm_thr))
            overlay_im[..., [1]] = np.maximum(overlay_im[..., [1]], gtCube)
            overlay_im[..., [2]] = np.maximum(overlay_im[..., [2]], VotRaw)

            fig, ax = plt.subplots(1, 2)
            fig.suptitle("Interactive Plot (Axial View)\n - Scroll wheel to navigate slices -")
            ax1, ax2 = ax.ravel()

            tracker1 = plotter3D.IndexTracker(ax1, np.repeat(imgCube, 3, axis=-1), 0)
            tracker2 = plotter3D.IndexTracker(ax2, overlay_im, 0)
            fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
            fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
            plt.show()"""



        # PRINT ========================================================================================================
        evaluations_global = dict([(k, (evaluations_global[k] + evaluations_currFold[k])) for k in evaluations_global])
        df_evaluations_currFold = pd.DataFrame(data=evaluations_currFold, index=None)
        print("Segmentation Evaluation:")
        print(df_evaluations_currFold)
        csv_path = f"scores/{script_name}_{experiment_name}.csv"
        if os.path.exists(csv_path):
            csv_mode = "a"
        else:
            csv_mode = "w"
        df_evaluations_currFold.to_csv(csv_path, mode=csv_mode, index=False)

        iteration += 1

    # PRINT AVERAGE ====================================================================================================

    df_evaluations_global = pd.DataFrame(data=evaluations_global, index=None)
    print("Segmentation Evaluation (All Records)")
    print(df_evaluations_global)
    df_evaluations_global.to_csv("scores/test_score_global2.csv", index=False)

    print("Segmentation Evaluation (Averaged)")
    df_evaluations_global = df_evaluations_global.set_index(['Method', 'Protocol', 'Setting', 'Threshold'])
    print(df_evaluations_global.groupby(['Method', 'Protocol', 'Setting', 'Threshold']).mean())
