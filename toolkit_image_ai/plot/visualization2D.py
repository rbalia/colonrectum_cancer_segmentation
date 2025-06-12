import matplotlib.pyplot as plt
import numpy as np

def dinamicFigurePlot(maintitle, figuretitles, images, shape, figureylabels=[], cmaps=None, figsize=(10, 10),
                      mode="show", figure_name="", sharex=False, sharey=False):
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
        err = 2
        # if image exist, show image
        if i < len(images):
            ax.imshow(images[i], cmap=cmaps[i])
            err -= 1

        # if title exist, print
        if i < len(figuretitles):
            ax.set_title(figuretitles[i])
            err -= 1

        # if ylabel exist, and image slot is in the first column, print ylabel
        if i % shape[1] == 0:
            if i//shape[1] < len(figureylabels):
                ax.set_ylabel(figureylabels[i//shape[1]])
            err -= 1

        # if multiple errors occurs, hide slot
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


def predColorVisualization(img, mask, pred):
    rgb = np.dstack(
            (img - (mask * 0.4) + (pred * 0.4),
             img - (pred * 0.4) + (mask * 0.4),
             img - (pred * 0.4) - (mask * 0.4)))
    rgb[rgb>1]=1.
    rgb[rgb<0]=0
    return rgb