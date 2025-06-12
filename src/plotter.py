import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import numpy as np


def printSlicBrief(original, boundaries_openCV, boundaries_Skimage, AVG_openCV, AVG_Skimage, ):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Superpixel Segmentantation")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(original)
    ax1.set_title('Original Image')

    ax2.imshow(boundaries_openCV)
    ax2.set_title('SLIC Segmentation (OpenCV)')

    ax3.imshow(AVG_openCV)
    ax3.set_title('AverageColor - (OpenCV)')

    ax4.imshow(original)
    ax4.set_title('Original Image')

    ax5.imshow(boundaries_Skimage)
    ax5.set_title('SLIC Segmentation (Skimage)')

    ax6.imshow(AVG_Skimage)
    ax6.set_title('AverageColor - (Skimage)')

    plt.show()


def printProcessingBrief(slicSegm, EdgeMap, newSlicSegm, labels, edgelessLabels, bordersEnhance):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Processing Phase")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(slicSegm)
    ax1.set_title('Slic Segmentation')

    ax2.imshow(EdgeMap)
    ax2.set_title('Edge Map')

    ax3.imshow(newSlicSegm)
    ax3.set_title('Edgeless Segmentation')

    ax4.imshow(labels, cmap="gray")
    ax4.set_title('Labels')

    ax5.imshow(edgelessLabels, cmap="gray")
    ax5.set_title('Edgeless Labels')

    ax6.imshow(bordersEnhance)
    ax6.set_title('New Segments Borders')

    plt.show()


def printBrief3Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    plt.show()


def printBrief4Cells(title, names, images, shape=(2, 2)):
    # Display result
    fig2, ax_arr = plt.subplots(shape[0], shape[1], sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    plt.show()


def printBrief6Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(images[0], cmap="gray")
    ax1.set_title(names[0])

    ax2.imshow(images[1], cmap="gray")
    ax2.set_title(names[1])

    ax3.imshow(images[2], cmap="gray")
    ax3.set_title(names[2])

    ax4.imshow(images[3], cmap="gray")
    ax4.set_title(names[3])

    ax5.imshow(images[4], cmap="gray")
    ax5.set_title(names[4])

    ax6.imshow(images[5], cmap="gray")
    ax6.set_title(names[5])

    plt.show()


def printBrief8Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax_arr.ravel()

    ax1.imshow(images[0], vmin=0, vmax=1)
    ax1.set_title(names[0])

    ax2.imshow(images[1], vmin=0, vmax=1)
    ax2.set_title(names[1])

    ax3.imshow(images[2], vmin=0, vmax=1)
    ax3.set_title(names[2])

    ax4.imshow(images[3], vmin=0, vmax=1)
    ax4.set_title(names[3])

    ax5.imshow(images[4], vmin=0, vmax=1)
    ax5.set_title(names[4])

    ax6.imshow(images[5], vmin=0, vmax=1)
    ax6.set_title(names[5])

    ax7.imshow(images[6], vmin=0, vmax=1)
    ax7.set_title(names[6])

    ax8.imshow(images[7], vmin=0, vmax=1)
    ax8.set_title(names[7])

    plt.show()

def predColorVisualization(img, mask, pred):
    rgb = np.dstack(
            (img - (mask * 0.4) + (pred * 0.4),
             img - (pred * 0.4) + (mask * 0.4),
             img - (pred * 0.4) - (mask * 0.4)))
    rgb[rgb>1]=1.
    rgb[rgb<0]=0
    return rgb

def dinamicFigurePlot(maintitle, figuretitles, images, shape, cmaps=None, figsize=(10, 10), mode="show",
                      figure_name="", sharex=False, sharey=False):
    #if len(images) != len(figuretitles):
    #    print("Warning: figure titles count do not match the number of images")
    if cmaps is None:
        cmaps = len(images) * [None]
    elif len(cmaps) < len(images):
        cmaps = cmaps + (len(images) - len(cmaps)) * [None]

    fig, ax_arr = plt.subplots(shape[0], shape[1], sharex=sharex, sharey=sharey, figsize=figsize)
    fig.suptitle(maintitle)

    for i, ax in enumerate(ax_arr.ravel()):
        #print(i)
        ax.axis('off')
        err = 2
        if i < len(images):
            ax.imshow(images[i], cmap=cmaps[i])
            err -= 1

        if i < len(figuretitles):
            ax.set_title(figuretitles[i])
            err -= 1

        if err == 2:
            ax.set_visible(False)


    if mode == "show":
        plt.show()
    elif mode == "save":
        plt.savefig(f'{figure_name}.png', dpi=fig.dpi)
        plt.close()
    else:
        print("Warning: invalid mode argument")
        plt.close()


def printBrief9Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    ax9.imshow(images[8])
    ax9.set_title(names[8])

    plt.show()


def printBrief10Cells(title, names, images, fig_name):
    # Display result
    fig2, ax_arr = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(15, 9))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    ax9.imshow(images[8])
    ax9.set_title(names[8])

    ax10.imshow(images[9])
    ax10.set_title(names[9])

    # plt.show()
    plt.savefig(f'{fig_name}.png', dpi=fig2.dpi)
    plt.close()


def printBrief12Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    ax9.imshow(images[8])
    ax9.set_title(names[8])

    ax10.imshow(images[9])
    ax10.set_title(names[9])

    ax11.imshow(images[10])
    ax11.set_title(names[10])

    ax12.imshow(images[11])
    ax12.set_title(names[11])

    plt.show()


def printBrief15Cells(title, names, images):
    # Display result
    fig2, ax_arr = plt.subplots(3, 5, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(title)
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15 = ax_arr.ravel()

    ax1.imshow(images[0])
    ax1.set_title(names[0])

    ax2.imshow(images[1])
    ax2.set_title(names[1])

    ax3.imshow(images[2])
    ax3.set_title(names[2])

    ax4.imshow(images[3])
    ax4.set_title(names[3])

    ax5.imshow(images[4])
    ax5.set_title(names[4])

    ax6.imshow(images[5])
    ax6.set_title(names[5])

    ax7.imshow(images[6])
    ax7.set_title(names[6])

    ax8.imshow(images[7])
    ax8.set_title(names[7])

    ax9.imshow(images[8])
    ax9.set_title(names[8])

    ax10.imshow(images[9])
    ax10.set_title(names[9])

    ax11.imshow(images[10])
    ax11.set_title(names[10])

    ax12.imshow(images[11])
    ax12.set_title(names[11])

    ax13.imshow(images[12])
    ax13.set_title(names[12])

    ax14.imshow(images[13])
    ax14.set_title(names[13])

    ax15.imshow(images[14])
    ax15.set_title(names[14])
    plt.show()


def printRoiBrief(roi, clinicalMask, clinicalRoi, highlightedRegions):
    # Display result
    fig2, ax_arr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Processing Phase")
    ax1, ax2, ax3, ax4 = ax_arr.ravel()

    ax1.imshow(roi)
    ax1.set_title('Detected ROI')

    ax2.imshow(clinicalMask)
    ax2.set_title('Clinical Mask')

    ax3.imshow(clinicalRoi)
    ax3.set_title('Selected ROI from Clinical Mask')

    ax4.imshow(highlightedRegions)
    ax4.set_title('Highlighted Regions')

    plt.show()


def printClassificationBrief(img, segmentation, roi, clinicalRoi, clinicalMask, prediction):
    # Display result
    fig2, ax_arr = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Classification Phase")
    ax1, ax2, ax3, ax4, ax5, ax6 = ax_arr.ravel()

    ax1.imshow(img, cmap="gray")
    ax1.set_title('Original Image')

    ax2.imshow(segmentation, cmap="gray")
    ax2.set_title('Segmentation')

    ax3.imshow(roi, cmap="gray")
    ax3.set_title('ROI')

    ax4.imshow(clinicalRoi, cmap="gray")
    ax4.set_title('Clinical ROI')

    ax5.imshow(clinicalMask, cmap="gray")
    ax5.set_title('Clinical Mask')

    ax6.imshow(prediction, cmap="gray")
    ax6.set_title('Prediction')

    plt.show()


def printClassificationCompareBrief(image, predictionMask_LR, predictionMask_LDA, predictionMask_KNN,
                                    clinicalMask, predictionMask_CART, predictionMask_NB, predictionMask_SVM):
    # Display result
    fig2, ax_arr = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle("Classification Phase")
    ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8 = ax_arr.ravel()

    ax1.imshow(image, cmap="gray")
    ax1.set_title('Original Image')

    ax2.imshow(predictionMask_LR, cmap="gray")
    ax2.set_title('LR')

    ax3.imshow(predictionMask_LDA, cmap="gray")
    ax3.set_title('LDA')

    ax4.imshow(predictionMask_KNN, cmap="gray")
    ax4.set_title('KNN')

    ax5.imshow(clinicalMask, cmap="gray")
    ax5.set_title('Clinical Mask')

    ax6.imshow(predictionMask_CART, cmap="gray")
    ax6.set_title('CART')

    ax7.imshow(predictionMask_NB, cmap="gray")
    ax7.set_title('NB')

    ax8.imshow(predictionMask_SVM, cmap="gray")
    ax8.set_title('SVM')

    plt.show()


def printHistogram(image):
    if len(image.shape) == 3:
        plt.subplot(121), plt.imshow(image)
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv.calcHist([image], [i], None, [256], [0, 256])
            plt.subplot(122), plt.plot(histr, color=col)
    else:
        plt.subplot(121), plt.imshow(image, 'gray')
        plt.subplot(122), plt.hist(image.ravel(), 256, [0, 256])

    plt.xlim([0, 256])
    plt.show()



def printTrainingBrief(history, modelName):
    # Display result
    fig2, ax_arr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 10))
    fig2.suptitle(modelName)
    ax1, ax2 = ax_arr.ravel()

    ax1.set_title('Accuracy Monitor')
    ax1.plot(history.history['accuracy'], label='train acc')
    ax1.plot(history.history['val_accuracy'], label='val acc')
    ax1.legend()

    ax2.set_title('Loss Monitor')
    ax2.plot(history.history['loss'], label='train loss')
    ax2.plot(history.history['val_loss'], label='val loss')
    ax2.legend()

    plt.show()


def plotSegmentationHistory(history, modelName, plotFigures=False, mode="save", save_path="plot_segm.jpg"):
    #print(history.history.keys())

    fig, ax_arr = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15, 10))
    fig.suptitle(modelName)
    ax1, ax2 = ax_arr.ravel()

    # summarize history for accuracy
    ax1.plot(history.history["iou_coef"], "r.-")
    ax1.plot(history.history["dice_coef"], "g.-")
    # plt.plot(history.history["accuracy"], "b.-")
    ax1.plot(history.history["val_iou_coef"], "r+-")
    ax1.plot(history.history["val_dice_coef"], "g+-")
    # plt.plot(history.history["val_accuracy"], "b+-")
    ax1.set_title(f"{modelName}  metrics")
    ax1.set_ylabel('Metrics')
    ax1.set_xlabel('Epochs')
    ax1.legend(["train iou", 'train dice',  # "train accuracy",
                "val iou", "val dice",  # "validation accuracy"
                ], loc='lower right')

    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f"{modelName}  loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylim(0,np.mean(history.history['val_loss'])*2)
    ax2.legend(['train', 'validation'], loc='upper right')
    # plt.savefig("trainingFigureSegmentation/" + modelName + "_" + category + "loss")

    if plotFigures and mode=="show":
        plt.show()
    elif mode=="save":
        plt.savefig(save_path)
        plt.close()
    #plt.clf()

def plotClassificationTraining(history, modelName, category, plotFigures=False):
    print(history.history.keys())

    # summarize history for accuracy
    plt.plot(history.history["accuracy"], "g.-")
    plt.plot(history.history["val_accuracy"], "r+-")
    plt.title(modelName + ' Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(["train accuracy", "validation accuracy"], loc='lower right')
    # plt.savefig("trainingFigureClassification/"+modelName+"_"+category+"metrics")
    if plotFigures:
        plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(modelName + ' loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'validation'], loc='upper right')
    # plt.savefig("trainingFigureClassification/" + modelName + "_" + category + "loss")
    if plotFigures:
        plt.show()
    plt.clf()
