import time
from time import sleep

import pydicom
import numpy as np
from skimage import transform, img_as_ubyte
from skimage.transform import resize, warp_polar
from sklearn.utils import shuffle

import plotter3D

def normalize(arr):
    if np.count_nonzero(arr)>0:
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)
    else:
        return arr

def load_dicom(dicom_path, resize=False, new_shape=None, return_dcm=False):
    """Load a Dicom File, preprocess, and return as numpy array.

        This function take a dicom path and
        Arguments:

    """
    # Load Dicom cube
    dcm_img = pydicom.dcmread(dicom_path)

    # Read spacing (sampling) information
    h_spacing = float(dcm_img.PixelSpacing[0])
    w_spacing = float(dcm_img.PixelSpacing[1])
    try:
        s_spacing = float(dcm_img.PixelSpacing[2])
    except:
        try:
            s_spacing = float(dcm_img.SliceThickness)
        except:
            # TODO: In caso di totale assenza di informazioni sulla paziatura tra le slice, prende una delle altre
            #  (Da gestire)
            s_spacing = h_spacing

    spacing = [s_spacing, h_spacing, w_spacing]

    # Get array
    us_img = dcm_img.pixel_array

    # Transpose: [Slices, Height, Width] -> [Height, Width, Slices]
    #us_img = np.transpose(us_img, (1, 2, 0))

    # Get unresized shape of dicom cube
    dcm_shape = us_img.shape

    # Resize image
    if resize:
        us_img = transform.resize(us_img, new_shape, mode='constant')

    # Normalize
    us_img = normalize(us_img)

    # Expand array with channel axis
    us_img = np.expand_dims(us_img, axis=-1)

    if return_dcm:
        return us_img, {"shape": dcm_shape, "spacing": spacing}, dcm_img
    else:
        return us_img, {"shape": dcm_shape, "spacing": spacing}


def overlay_prediction_to_dicom(dcm_img, msk_les_rgb, msk_nod_rgb, resize_shape=None, verbose=False):

    if verbose: print("Start DICOM Labelling...")
    start = time.time()

    # Masks Preprocessing
    if resize_shape is not None:
        msk_les_rgb = resize(msk_les_rgb, (96,160,160,3), order=0, anti_aliasing=False)
        msk_nod_rgb = resize(msk_nod_rgb, (96,160,160,3), order=0, anti_aliasing=False)

    # Dicom Preprocessing
    if verbose: print("Preprocessing Dicom Image")
    pixel_array = dcm_img.pixel_array

    # Normalize values
    pixel_array = normalize(pixel_array)

    # Add Channel axis and repete
    pixel_array = np.expand_dims(pixel_array, axis=-1)
    pixel_array_rgb = np.repeat(pixel_array, 3, axis=-1)

    # MASK PREPROCESSING ===
    if verbose: print("Preprocessing Generated Masks")
    # Transpose to [Slice, H, W, Ch]
    #msk_les_rgb = np.transpose(mask_les_rgb, (2, 0, 1, 3))
    #msk_nod_rgb = np.transpose(mask_nod_rgb, (2, 0, 1, 3))

    #msk_les_rgb_i = np.copy(msk_les_rgb)
    #msk_nod_rgb_i = np.copy(msk_nod_rgb)

    msk_les_rgb_i = (1 - msk_les_rgb) * (msk_les_rgb)
    msk_nod_rgb_i = (1 - msk_nod_rgb) * (msk_nod_rgb)

    # Combine Color Information and resize to dicom shape
    if verbose: print("Overlaying Predictions to Image")

    cs = 0.5
    csi = cs * (-1)

    # Combine multiple rgb-mask with color saturation enhancement
    msk_color_comb = (msk_les_rgb_i * csi) + (msk_nod_rgb_i * csi) + (msk_les_rgb * cs) + (msk_nod_rgb * cs)
    msk_color_comb = resize(msk_color_comb, pixel_array_rgb.shape, mode='constant')#, order=0, anti_aliasing=False)

    # Apply Overlay
    pixel_array_rgb_overlaid = pixel_array_rgb + msk_color_comb

    # Clip
    pixel_array_rgb_overlaid[pixel_array_rgb_overlaid > 1] = 1
    pixel_array_rgb_overlaid[pixel_array_rgb_overlaid < 0] = 0

    if verbose: print(f"Elapsed Time - DICOM Annotation: {time.time() - start} seconds")


    return pixel_array_rgb_overlaid


def compute_subcubes_count(cube_size, subcube_size, stride):
    num_subcubes = ((cube_size[0] - subcube_size[0]) // stride + 1) * \
                   ((cube_size[1] - subcube_size[1]) // stride + 1) * \
                   ((cube_size[2] - subcube_size[2]) // stride + 1)
    #print(f"Numero sottocubi estraibili: {num_subcubes}")
    return num_subcubes

def extract_subcubes(X, subcube_size, stride):

    # Calcoliamo il numero totale di sottocubi che possiamo estrarre dall'immagine
    num_subcubes = compute_subcubes_count(X.shape, subcube_size, stride)

    # Inizializziamo un array per contenere i sottocubi estratti
    subcubes = np.zeros((num_subcubes,) + subcube_size)

    # Estraiamo i sottocubi utilizzando la finestra scorrevole
    subcube_idx = 0
    for i in range(0, X.shape[0] - subcube_size[0] + 1, stride):
        for j in range(0, X.shape[1] - subcube_size[1] + 1, stride):
            for k in range(0, X.shape[2] - subcube_size[2] + 1, stride):
                subcube = X[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :]
                subcubes[subcube_idx, :, :, :, :] = subcube
                subcube_idx += 1
    return subcubes

def reconstruct_image(subcubes, image_shape, subcube_size, stride, mode="avg"):
    """
    Ricostruisce l'immagine originale a partire dai sottocubi estratti.

    Args:
        subcubes (ndarray): array contenente i sottocubi estratti dall'immagine originale
        image_shape (tuple): tupla contenente le dimensioni dell'immagine originale
        subcube_size (tuple): tupla contenente le dimensioni dei sottocubi
        stride (int): passo di spostamento della finestra scorrevole utilizzata per estrarre i sottocubi

    Returns:
        ndarray: array contenente l'immagine ricostruita
    """
    # Inizializza l'immagine ricostruita con zeri
    image = np.zeros(image_shape)

    # Inizializza un array che conta il numero di sovrapposizioni di ogni voxel
    overlap_count = np.zeros(image_shape)
    subcube_idx = 0
    # Scorri lungo i sottocubi e aggiungi ogni sottocubo all'immagine ricostruita
    for i in range(0, image_shape[0] - subcube_size[0] + 1, stride):
        for j in range(0, image_shape[1] - subcube_size[1] + 1, stride):
            for k in range(0, image_shape[2] - subcube_size[2] + 1, stride):
                subcube = subcubes[subcube_idx, :, :, :, :]
                if mode=="avg":
                    image[i:i+subcube_size[0], j:j+subcube_size[1], k:k+subcube_size[2], :] += subcube
                    overlap_count[i:i+subcube_size[0], j:j+subcube_size[1], k:k+subcube_size[2], :] += 1
                elif mode=="max":
                    image[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :] = \
                        np.maximum(image[i:i + subcube_size[0], j:j + subcube_size[1], k:k + subcube_size[2], :],subcube)
                subcube_idx += 1

    # Divide ogni voxel dell'immagine ricostruita per il numero di sovrapposizioni per ottenere la media
    if mode == "avg":
        image /= overlap_count

    return image

def reconstruct_dataset_from_subs(X, Y, predictions, cube_shape, subs_shape, stride):
    X_rec = []
    Y_rec = []
    predictions_rec = np.empty(len(predictions), dtype=object)  # ([]*len(predictions))

    n_subs = compute_subcubes_count(cube_shape, subs_shape, stride)
    for i in range(0, len(X), n_subs):
        img_reconstruction = reconstruct_image(X[i:i + n_subs], cube_shape, subs_shape, stride)
        gt_reconstruction = reconstruct_image(Y[i:i + n_subs], cube_shape, subs_shape, stride)
        for j in range(0, len(predictions)):
            if predictions_rec[j] is None:
                predictions_rec[j] = []
            pred_reconstruction = reconstruct_image(predictions[j][i:i + n_subs], cube_shape,
                                                    subs_shape, stride, mode="avg")
            predictions_rec[j].append(pred_reconstruction)
        X_rec.append(img_reconstruction)
        Y_rec.append(gt_reconstruction)
        # preds_rec.append(pred_reconstruction)
    X_rec = np.array(X_rec)
    Y_rec = np.array(Y_rec)

    # Convert from array of object to array of float
    predictions_rec = np.array(list(predictions_rec), dtype=np.float64)
    #np.array(predictions_rec)
    #np.vstack(predictions_rec[:]).astype(np.float64))

    return X_rec, Y_rec, predictions_rec

def count_ditribution(X,Y, verbose=True):
    valid_cubes = 0
    empty_cubes = 0

    for x, y in zip(X, Y):
        if np.count_nonzero(y) > 10:
            valid_cubes += 1
        else:
            empty_cubes += 1

    if verbose:
        print(f"Empty Cubes: {empty_cubes} - Valid Cubes: {valid_cubes}")
    return empty_cubes, valid_cubes

def split_valid_empty(X,Y, verbose=False):
    X_filt_valid = []
    Y_filt_valid = []
    X_filt_empty = []
    Y_filt_empty = []
    valid_cubes = 0
    empty_cubes = 0

    empty_cubes_pre, valid_cubes_pre = count_ditribution(X,Y, verbose=verbose)

    if verbose:
        print(f"(PreFilter) Empty Cubes: {empty_cubes_pre} - Valid Cubes: {valid_cubes_pre}")

    i = 0
    for x,y in zip(X,Y):
        if np.count_nonzero(y)>10:
            X_filt_valid.append(x)
            Y_filt_valid.append(y)
            #plotter3D.plotVolumetricSlices(x, [y], axis_name=["Ax", "Cor", "Sag"],
            #                               mask_mean_projection=True,
            #                               figure_title=f"VALID {i}: {np.count_nonzero(y)}")
            valid_cubes+=1
        else:
            X_filt_empty.append(x)
            Y_filt_empty.append(y)
            #plotter3D.plotVolumetricSlices(x, [y], axis_name=["Ax", "Cor", "Sag"],
            #                               mask_mean_projection=True,
            #                               figure_title=f"EMPTY {i}: {np.count_nonzero(y)}")
            empty_cubes +=1
        i+=1

    if verbose:
        print(f"(PostFilter) Empty Cubes: {empty_cubes} - Valid Cubes: {valid_cubes}")
    #X_filt_valid = np.array(X_filt_valid)
    #Y_filt_valid = np.array(Y_filt_valid)
    #X_filt_empty = np.array(X_filt_empty)
    #Y_filt_empty = np.array(Y_filt_empty)
    return X_filt_valid, Y_filt_valid, X_filt_empty, Y_filt_empty

def filter_empty_cubes(X,Y, rate=None, verbose=False, shuffle_flag=False):
    X_filt = []
    Y_filt = []
    valid_cubes = 0
    empty_cubes = 0

    empty_cubes_pre, valid_cubes_pre = count_ditribution(X,Y, verbose=verbose)

    if verbose:
        print(f"(PreFilter) Empty Cubes: {empty_cubes_pre} - Valid Cubes: {valid_cubes_pre}")
    if valid_cubes_pre == 0:
        return X_filt, Y_filt
    if rate is None:
        rate = empty_cubes_pre//valid_cubes_pre
        if rate == 0:
            rate = 1
    i=0
    if shuffle_flag:
        X, Y = shuffle(X,Y)
    for x,y in zip(X,Y):
        if np.count_nonzero(y)>5:
            X_filt.append(x)
            Y_filt.append(y)
            valid_cubes+=1
        elif i%rate == 0:
            X_filt.append(x)
            Y_filt.append(y)
            empty_cubes +=1
        i+=1

    if verbose:
        print(f"(PostFilter) Empty Cubes: {empty_cubes} - Valid Cubes: {valid_cubes}")
    X_filt = np.array(X_filt)
    Y_filt = np.array(Y_filt)
    return X_filt, Y_filt

def split_empty_masks(X,Y):
    X_filt = []
    Y_filt = []
    valid_cubes = 0
    empty_cubes = 0

    i=0
    for x,y in zip(X,Y):
        if np.count_nonzero(y)>0:
            X_filt.append(x)
            Y_filt.append(y)
            valid_cubes+=1
        else:
            X_filt.append(x)
            Y_filt.append(y)
            empty_cubes +=1
        i+=1

    print(f"Empty Cubes: {empty_cubes} - Valid Cubes: {valid_cubes}")
    X_filt = np.array(X_filt)
    Y_filt = np.array(Y_filt)
    return X_filt, Y_filt

def project_circumference(mask):
    slice_projection = np.amax(mask, axis=0)
    polar_projection = warp_polar(slice_projection, output_shape=slice_projection.shape)
    circm_projection = np.amax(polar_projection, axis=1)
    circm_projection = np.expand_dims(circm_projection, axis=-1)
    return slice_projection, polar_projection, circm_projection


def mass_ends_clock_position(circumference_projection, circumference_involved=0):
    # Compute Circumference involved as hours limits
    len_proj = len(circumference_projection)
    if np.count_nonzero(circumference_projection) == len_proj:
        lesion_start = 0
        lesion_stop = 360
    else:
        lesion_start = -1
        lesion_stop = -1
        id = 0
        while lesion_stop == -1 or lesion_start == -1:
            if circumference_projection[id] == 0 and circumference_projection[(id + 1) % len_proj] > 0:
                lesion_start = id + 1
            elif circumference_projection[id] > 0 and circumference_projection[(id + 1) % len_proj] == 0:
                lesion_stop = id
            elif np.count_nonzero(circumference_projection) == len_proj:
                lesion_start = 0
                lesion_stop = 0
            id = (id + 1) % len_proj

        # [0-1] to [0-360], Zero is located on right (3 o'clock)
        lesion_start = (lesion_start / len_proj) * 360
        lesion_stop = (lesion_stop / len_proj) * 360

    # Degree to Hour Conversion (Add a shift of 90 degrees)
    lesion_start = round(((lesion_start+ 90) % 360)/30,1)
    lesion_stop  = round(((lesion_stop+ 90) % 360)/30,1)

    return lesion_start, lesion_stop

