import matplotlib.pyplot as plt
import numpy as np

def plotSegmentationHistory(history, modelName, plotFigures=False, mode="save", save_path="plot_segm.jpg"):
    #print(history.history.keys())

    fig, ax_arr = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(15, 10))
    fig.suptitle(modelName)
    ax1, ax2 = ax_arr.ravel()

    # summarize history for accuracy
    ax1.plot(history["iou_coef"], "r.-")
    ax1.plot(history["dice_coef"], "g.-")
    # plt.plot(history.history["accuracy"], "b.-")

    ax1.plot(history["val_iou_coef"], "r+-")
    ax1.plot(history["val_dice_coef"], "g+-")
    # plt.plot(history.history["val_accuracy"], "b+-")

    ax1.set_title(f"{modelName}  metrics")
    ax1.set_ylabel('Metrics')
    ax1.set_xlabel('Epochs')
    ax1.legend(["train iou", 'train dice',  # "train accuracy",
                "val iou", "val dice",  # "validation accuracy"
                ], loc='lower right')

    # summarize history for loss
    ax2.plot(history['loss'])
    ax2.plot(history['val_loss'])
    ax2.set_title(f"{modelName}  loss")
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylim(0, np.mean(history['val_loss'])*2)
    ax2.legend(['train', 'validation'], loc='upper right')
    # plt.savefig("trainingFigureSegmentation/" + modelName + "_" + category + "loss")

    if plotFigures and mode=="show":
        plt.show()
    elif mode=="save":
        plt.savefig(save_path)
        plt.close()
    else:
        print("Not a valid mode has been selected")
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
