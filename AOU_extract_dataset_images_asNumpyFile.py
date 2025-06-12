import csv
import gc
import glob
import os
import re
from re import match
from time import sleep

import cv2
import nrrd
import pandas as pd
import SimpleITK as sitk
import pydicom
import numpy as np
from mpl_toolkits import mplot3d
from stl import mesh
import stltovoxel
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from stltovoxel import convert_meshes

import plotter3D
from src import plotter
from plotter3D import *
from skimage.transform import resize
import ipyvolume


if __name__ == '__main__':

    dataset_us = []
    dataset_p_gt = []
    dataset_l_gt = []

    dataset_infos = []

    dataset_dir = "dataset_colonrectum/raw_data_CA"
    dataset_pkgs_dst = "dataset_colonrectum/binaryPkg_CA_postCorrezione"

    caseID = 0
    for index, case_dir in enumerate(os.listdir(dataset_dir)):
        if case_dir[0] == ".":
            continue
        case_path = dataset_dir + "/" + case_dir + "/"
        ultrasound_path = case_path #+ case_dir + ".dcm"
        neoplasia_nrrd_path = case_path + case_dir + "_neoplasia.nrrd"
        lymphnode_nrrd_path = case_path + case_dir + "_linfonodo.nrrd"
        print(f"Processing case: {case_path}")

        located_dcm = glob.glob(ultrasound_path+"*.dcm")[0]
        dcm_img = pydicom.dcmread(located_dcm)
        us_img = dcm_img.pixel_array

        # Read NRRD File
        neoplasia_gt, neoplasia_gt_header = nrrd.read(neoplasia_nrrd_path)
        try:
            lymphnode_gt, lynphnode_gt_header = nrrd.read(lymphnode_nrrd_path)
        except:
            lymphnode_gt = np.zeros_like(neoplasia_gt)

        # Rotate Ground Truths
        neoplasia_gt = np.rot90(neoplasia_gt, k=-1, axes=(0,1))
        neoplasia_gt = np.flip(neoplasia_gt, axis=1)
        neoplasia_gt = np.transpose(neoplasia_gt, (2, 0, 1))

        lymphnode_gt = np.rot90(lymphnode_gt, k=-1, axes=(0, 1))
        lymphnode_gt = np.flip(lymphnode_gt, axis=1)
        lymphnode_gt = np.transpose(lymphnode_gt, (2, 0, 1))

        # Normalize image and masks
        us_img = normalize(us_img)
        neoplasia_gt = normalize(neoplasia_gt)
        lymphnode_gt = normalize(lymphnode_gt)

        # Resize image (with nearest-neighbour, in order to get a resized binary image)
        new_shape = (160, 160, 160)
        us_img = resize(us_img, new_shape, mode='constant')
        neoplasia_gt = resize(neoplasia_gt, new_shape, order=0, anti_aliasing=False)
        lymphnode_gt = resize(lymphnode_gt, new_shape, order=0, anti_aliasing=False)

        # Expand dimension (Add Channel Axis)
        us_img = np.expand_dims(us_img, axis=-1)
        neoplasia_gt = np.expand_dims(neoplasia_gt, axis=-1)
        lymphnode_gt = np.expand_dims(lymphnode_gt, axis=-1)

        # Export as Numpy Array
        ultrasound_npy_path = f"{dataset_pkgs_dst}/single_us/ultrasounds_{caseID:03d}.npy"
        neoplasia_npy_path = f"{dataset_pkgs_dst}/single_neoplasia_gt/neoplasia_gts_{caseID:03d}.npy"
        lymphnode_npy_path = f"{dataset_pkgs_dst}/single_lymphnode_gt/lymphnode_gts_{caseID:03d}.npy"

        np.save(ultrasound_npy_path, us_img)
        np.save(neoplasia_npy_path, neoplasia_gt)
        np.save(lymphnode_npy_path, lymphnode_gt)

        # Get T grade
        located_tgrade = glob.glob(ultrasound_path + "stadiazione*")[0]
        tgrade = located_tgrade[-5]
        if not tgrade.isdigit():
            tgrade = ""

        # Save dataset information to csv
        if not os.path.exists(neoplasia_nrrd_path):
            neoplasia_nrrd_path = ""
        if not os.path.exists(lymphnode_nrrd_path):
            lymphnode_nrrd_path = ""

        dataset_infos.append({
            "dicom": located_dcm,
            "image_npy": ultrasound_npy_path,
            "neoplasia_nrrd": neoplasia_nrrd_path,
            "lymphnode_nrrd": lymphnode_nrrd_path,
            "neoplasia_npy": neoplasia_npy_path,
            "lymphnode_npy": lymphnode_npy_path,
            "t_grade": tgrade,
            })

        print(dataset_infos)
        caseID += 1

        print(us_img.shape)
        print(neoplasia_gt.shape)
        print(lymphnode_gt.shape)
        #plotter3D.plotVolumetricSlices(us_img, [neoplasia_gt, lymphnode_gt], ["A","C","S"], mask_mean_projection=True)


    fieldnames = ["dicom", "image_npy", "neoplasia_nrrd", "lymphnode_nrrd", "neoplasia_npy", "lymphnode_npy","t_grade"]
    with open(f'{dataset_pkgs_dst}/dataset_paths.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_infos)

        #PLOT 3D =======================================================================================================
        """us_img_black = np.zeros(us_img.shape + (3,))
        us_img_black[...,0] = us_img[:,:,:]
        us_img_black[..., 1] = us_img[:, :, :]
        us_img_black[..., 2] = us_img[:, :, :]
        us_slice = us_img_black[:,:,:,:]
        us_img = us_slice
        print(us_slice.shape)
        plt.imshow(us_slice)"""

        #plt.show()
        #plotter.dinamicFigurePlot("",[f"{us_shape[0]//2},:,:",f":,{us_shape[1]//2},:",f":,:,{us_shape[2]//2}",
        #                              f"{us_shape[0]//2},:,:",f":,{us_shape[1]//2},:",f":,:,{us_shape[2]//2}",
        #                              f"{us_shape[0]//2},:,:",f":,{us_shape[1]//2},:",f":,:,{us_shape[2]//2}"],
        #                          [us_img[us_shape[0] // 2, :, :, :], us_img[:, us_shape[1] // 2, :, :], us_img[:, :, us_shape[2] // 2, :],
        #                           neoplasia_gt[us_shape[0]//2,:,:],neoplasia_gt[:,us_shape[1]//2,:],neoplasia_gt[:,:,us_shape[2]//2],
        #                           lymphnode_gt[us_shape[0]//2,:,:],lymphnode_gt[:,us_shape[1]//2,:],lymphnode_gt[:,:,us_shape[2]//2]],
        #                          shape=(3,3))


        """# Polar Coordinates
        img_tmp = us_img[:,:,0:1,0]
        im_size = us_img.shape[0]
        polar_img = cv2.warpPolar(img_tmp, (im_size, im_size),
                                  (img_tmp.shape[0] / 2, img_tmp.shape[1] / 2),
                                  img_tmp.shape[1], cv2.WARP_POLAR_LINEAR)
        # Rotate it sideways to be more visually pleasing
        polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plotter.dinamicFigurePlot("titolo", [], [img_tmp, polar_img], (1, 2))"""


        # Masked Plot
        #plt.imshow(us_img[:, us_shape[1] // 2, :], cmap='gray')  # I would add interpolation='none'
        #plt.imshow(neoplasia_gt[:,us_shape[1]//2,:], cmap='Reds', alpha=0.3 * (neoplasia_gt[:,us_shape[1]//2,:] > 0))
        #plt.show()

        """# Interactive 2D plot with scrolling
        fig, ax = plt.subplots(1, 2)
        ax1, ax2 = ax.ravel()
        tracker1 = IndexTracker(ax1, us_img, 4)
        tracker2 = IndexTracker(ax2, neoplasia_gt, 4)
        fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)
        fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)
        plt.show()"""

