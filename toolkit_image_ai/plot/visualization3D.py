import numpy as np
import toolkit_image_ai.plot.visualization2D
from toolkit_image_ai.plot import visualization2D


def plotVolumetricSlices(img, mask_list, axisX_name, axisY_name=[], img_mean_projection=False, mask_mean_projection=False,
                         merge_masks=False, transpose=False, figure_title=f"Centered Volumetric Slices"):
    img_sh = img.shape
    figure_rows = 1 + len(mask_list)
    #msk_sh = []
    mask_slices = []

    # Slice VMean Projection of each view
    if mask_mean_projection:
        for mask in mask_list:
            les_vol_0 = np.mean(mask, axis=0)
            les_vol_1 = np.mean(mask, axis=1)
            les_vol_2 = np.mean(mask, axis=2)
            if (np.max(mask)>0):
                les_vol_0 /= np.max(les_vol_0)
                les_vol_1 /= np.max(les_vol_1)
                les_vol_2 /= np.max(les_vol_2)
            mask_slices.append([les_vol_0, les_vol_1, les_vol_2])
    else:
        for mask in mask_list:
            msk_sh = mask.shape
            mask_slices.append([mask[msk_sh[0] // 2, :, :], mask[:, msk_sh[1] // 2, :], mask[:, :, msk_sh[2] // 2]])


    if img_mean_projection:
        img_vol_0 = np.mean(img, axis=0)
        img_vol_1 = np.mean(img, axis=1)
        img_vol_2 = np.mean(img, axis=2)
        img_vol_0 /= np.max(img_vol_0)
        img_vol_1 /= np.max(img_vol_1)
        img_vol_2 /= np.max(img_vol_2)
        img_slices = [img_vol_0, img_vol_1, img_vol_2]
    else:
        img_slices = [img[img_sh[0] // 2, :, :], img[:, img_sh[1] // 2, :], img[:, :, img_sh[2] // 2]]


    if merge_masks:
        mask_slices = np.array(mask_slices)
        mask_slices = np.sum(mask_slices, axis=0)
        for i, slice in enumerate(mask_slices):
            mask_slices[i] = np.clip(slice, 0, 1)
        mask_slices_ravel = mask_slices
        figure_rows = 2
    else:
        mask_slices_ravel = []
        for mask in mask_slices:
            mask_slices_ravel = mask_slices_ravel + mask

    if mask_mean_projection:
        figure_title += "\n(Le maschere sono mostrate come Proiezioni di Intensità Media)"

    axisY_names = axisY_name
    axisX_names = [f"{axisX_name[0]} (#{img_sh[0] // 2})",
                   f"{axisX_name[1]} (#{img_sh[1] // 2})",
                   f"{axisX_name[2]} (#{img_sh[2] // 2})"]
    images = [*img_slices, *mask_slices_ravel]
    figure_shape = (figure_rows, 3)
    cmaps = ["gray", "gray", "gray"]

    if transpose:
        figure_shape = (figure_shape[1], figure_shape[0])
        images_t = []
        cmaps_t = []

        for i in range(0, len(images)):
            images_t.append(images[(i * figure_shape[0]) % len(images)
                                   + ((i * figure_shape[0]) // len(images))])
            if i%figure_shape[1] == 0:
                cmaps_t.append("gray")
            else:
                cmaps_t.append(None)

        cmaps = cmaps_t
        images = images_t
        axis_Y_tmp = axisY_names
        axisY_names = axisX_names
        axisX_names = axis_Y_tmp

    visualization2D.dinamicFigurePlot(figure_title, axisX_names, images,
                                      figureylabels=axisY_names, shape=figure_shape, cmaps=cmaps)
