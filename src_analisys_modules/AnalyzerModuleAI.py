import shutil
import time

from skimage import img_as_ubyte

from plotter3D import *

import plotter3D
import json

#from utils import *

from src_analisys_modules import analysis
from src_analisys_modules import preprocessing
from src_analisys_modules import prediction
from skimage.transform import resize

class Logger:
    def __init__(self, path, verbose=False):
        self.path = path
        self.verbose = verbose
        self.logger_infos = {"macro_phase": "", "macro_progress": "",
                             "micro_phase": "","micro_progress": "",
                             "global_progress": 0}

    def update_progress(self):
        mac_p = self.logger_infos["macro_progress"].split("/")
        global_progress = int(mac_p[0]) / int(mac_p[1])
        if self.logger_infos["micro_progress"]:
            mic_p = self.logger_infos["micro_progress"].split("/")
            global_progress += (int(mic_p[0]) / int(mic_p[1])) * (1 / int(mac_p[1]))
        self.logger_infos["global_progress"] = int(global_progress*100)

        if self.verbose:
            print(f'\nLogger:')
            print(f'\tGlobal Progress: {self.logger_infos["global_progress"]}')
            print(f'\t\t{self.logger_infos["macro_progress"]} - {self.logger_infos["macro_phase"]}')
            print(f'\t\t{self.logger_infos["micro_progress"]} - {self.logger_infos["micro_phase"]}')
            print(f'\n')

    def write(self, macro_phase, macro_progress, micro_phase, micro_progress):
        if macro_phase is not None: self.logger_infos["macro_phase"] = macro_phase
        if macro_progress is not None: self.logger_infos["macro_progress"] = macro_progress
        if micro_phase is not None: self.logger_infos["micro_phase"] = micro_phase
        if micro_progress is not None: self.logger_infos["micro_progress"] = micro_progress
        self.update_progress()

        with open(self.path, "w") as log:
            json.dump(self.logger_infos, log)


class AnalyzerModuleAI:
    def __init__(self, dicom_path, lesion_mask_path=None, lymphnode_mask_path=None, t_stage=None):
        self.dicom_path = dicom_path
        self.gt_lesion_mask_path = lesion_mask_path
        self.gt_lymphnode_mask_path = lymphnode_mask_path
        self.gt_t_stage = t_stage

    def analyze(self, json_output_path="out.json", logger_path="log.json", dicom_output_path="out.dcm",
                return_analysis=True, use_gt=False, verbose=False, plot_figure=False):

        Log = Logger(logger_path, verbose=False)
        Log.write("Lettura del file DICOM", "0/7", "", "")

        if verbose: print("Loading DICOM Cube..")
        img, dcm_params, dcm_img = preprocessing.load_dicom(self.dicom_path, resize=False, return_dcm=True)

        # AI PREDICTION ================================================================================================
        if use_gt:
            # Load prediction masks (currently using ground truth)
            msk_les = np.load(self.gt_lesion_mask_path)
            msk_nod = np.load(self.gt_lymphnode_mask_path)
            t_stage = self.gt_t_stage

        # Use real predictions
        else:
            Log.write("Segmentazione della neoplasia", "1/7", "", "")


            # Lesion Segmentation - Method 2D
            #msk_les = prediction.detect_lesion_2D(img, segm_thr=0.5, img_shape=(160,160,1),
            #                                      weights_dir="model_weights/lesion_segmentation/2D_0/")

            # Lesion Segmentation - Method 3D
            msk_les = prediction.detect_lesion_3D_subcubes(img, Log, segm_thr=0.5,
                                                           cube_shape=(128,128,128,1), input_shape=(64,64,64,1),
                                                           weights_dir="model_weights/lesion_segmentation/3DSubs_0/",
                                                           verbose=verbose)

            #===========================================================================================================
            Log.write("Segmentazione dei linfonodi", "2/7", "", "")

            #msk_nod = prediction.detect_lymphnode_2D(img, segm_thr=0.5, input_shape=(160, 160, 1),
            #                                                  weights_dir="model_weights/lymphnode_segmentation/0/")


            msk_nod = prediction.detect_lymphnode_3D_subcubes(img, Log, segm_thr=0.5,
                                                              cube_shape=(160,160,160,1), input_shape=(64,64,64,1),
                                                              weights_dir="model_weights/lymphnode_segmentation/3DSubs_2/",
                                                              verbose=verbose)

            # ===========================================================================================================
            Log.write("Classificazione del grado di infiltrazione", "3/7", "", "")
            #t_stage, t_stage_confidence = prediction.estimate_infiltration_stage(img, Log,input_shape=(96, 96, 96, 1),
            #                                    verbose=verbose,weights_dir="model_weights/tstage_classification/0/")
            t_stage, t_stage_confidence = prediction.estimate_infiltration_stage_v2(img, Log, input_shape=(96,96,96,1),
                                                verbose=verbose, weights_dir="model_weights/tstage_classification/1/")
            alpha = 0.8
            model_fscore = 0.6
            final_confidence = (alpha * model_fscore) + ((1-alpha) * t_stage_confidence)
            final_confidence = round(final_confidence*100,2)

        # ANALYZE PREDICTIONS ==========================================================================================
        Log.write("Analisi e misurazione delle predizioni", "4/7", "", "")

        # Analyse Neoplasia Mass
        msk_les_label, msk_les_infos, mask_les_rgb = analysis.neoplasia_analysis(msk_les, dcm_params, verbose=verbose,
                                                                                 plot_circumference=False)

        # Analyse Lymph Nodes
        msk_nod_label, msk_nod_infos, mask_nod_rgb = analysis.lymphnode_analysis(msk_nod, dcm_params, verbose=verbose)


        # Overlay predictions to dicom image (return the modified dicom_img.pixel_array)
        Log.write("Sovraimpressione delle maschere nel Dicom", "5/7", "", "")

        pixel_array_rgb_overlaid = preprocessing.overlay_prediction_to_dicom(dcm_img, mask_les_rgb, mask_nod_rgb,
                                                                             resize_shape=(96,160,160,1), verbose=verbose)


        # PLOTTING =====================================================================================================
        im_plot_size = (96,160,160,3)

        if plot_figure:
            print("Start Plotting")
            im =  resize(img, (*im_plot_size[0:3],1), mode='constant')#transpose(img, (2,0,1,3))
            im_rgb = np.repeat(im,3,axis=-1)
            les = resize(mask_les_rgb, im_plot_size, order=0, anti_aliasing=False)#transpose(mask_les_rgb, (2, 0, 1, 3))
            nod = resize(mask_nod_rgb, im_plot_size, order=0, anti_aliasing=False)#transpose(mask_nod_rgb, (2, 0, 1, 3))

            # Volumetric Slices Plotting
            # Ordine assi atteso: [Slices, H, W, Ch]
            plotter3D.plotVolumetricSlices(im, [les,nod], ["Axial","Coronal","Sagittal"],
                                           img_mean_projection=False, mask_mean_projection=True, merge_masks=False)


            # Interactive Plotting
            # Ordine assi atteso: [Slices, H, W, Ch]
            imO = resize(pixel_array_rgb_overlaid, im_plot_size, mode='constant')
            join = np.maximum(les, nod)
            join = np.maximum(join, im_rgb*0.2)
            #join[join > 1] = 1.

            fig, ax = plt.subplots(1, 3)
            fig.suptitle("Interactive Plot (Axial View)\n - Scroll wheel to navigate slices -")
            ax1, ax2, ax3 = ax.ravel()
            tracker1 = plotter3D.IndexTracker(ax1, im_rgb, 0)
            tracker2 = plotter3D.IndexTracker(ax2, imO , 0)
            tracker3 = plotter3D.IndexTracker(ax3, join, 0)
            fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
            fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
            fig.canvas.mpl_connect('scroll_event', tracker3.on_scroll)
            plt.show()

        # GENERATE AND SAVE RESULTS ====================================================================================
        Log.write("Esportazione delle misurazioni", "6/7", "", "")

        if return_analysis:
            print("Exporting Results..")
            # Convert to 8bit integers
            pixel_array_rgb_overlaid = img_as_ubyte(pixel_array_rgb_overlaid)

            # Create overlaid dicom
            print("\tConvert and store DICOM")
            dcm_img.SamplesPerPixel = 3
            dcm_img.PlanarConfiguration = 0
            dcm_img.PhotometricInterpretation = "RGB"
            dcm_img.PixelData = pixel_array_rgb_overlaid.tobytes()
            dcm_img.save_as(dicom_output_path, write_like_original=False)

            # Compose JSON fields
            analysis_result = {
                "dicom_input_path": self.dicom_path,
                "dicom_output_path": dicom_output_path,
                "dicom_sampling": dcm_params["spacing"],
                "dicom_shape": dcm_params["shape"],
                "lesions": msk_les_infos,
                "lymphnodes": msk_nod_infos,
                "t_stage": t_stage,
                "confidence": final_confidence
            }

            print("\tStore JSON")
            with open(json_output_path, "w") as file:
                json.dump(analysis_result, file)

            Log.write("Analisi Completata", "7/7", "", "")