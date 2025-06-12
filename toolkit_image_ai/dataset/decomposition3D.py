import numpy as np


""" Collection of functions for managing 3D datasets and performing sliding window decomposition. """

def datagenTo3DArray(datagen):
    """ Convert the output of a DataGenerator to an array
        - Datagen in input should return fullcubes,
        - output an array formated as [Samples, Slices, Width, Height, Channels]
        then it is possible to call array3DToSubCubes for decomposition
    """
    X = []
    Y = []
    for i, ds in enumerate(datagen):
        img = ds[0][0]
        msk = ds[1][0]
        X.append(img)
        Y.append(msk)

    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def array3DToSubCubes(X, Y, subcube_size, stride):
    """ Convert a dataset array of fullbubes to an array of subcubes """
    X_sub = []
    Y_sub = []
    for x,y in zip(X,Y):
        a = extract_subcubes(x, subcube_size, stride)
        b = extract_subcubes(y, subcube_size, stride)

        for im, msk in zip(a,b):
            X_sub.append(im)
            Y_sub.append(msk)

    X_sub = np.array(X_sub)
    Y_sub = np.array(Y_sub)
    return X_sub, Y_sub

def datagenTo2DArray(datagen, slice_axis=0):
    """ Convert the output of a datagenerator to array of 2d slices """
    X = []
    Y = []
    for i, ds in enumerate(datagen):
        img = ds[0][0]
        msk = ds[1][0]
        for slice_id in range(0, img.shape[slice_axis]):
            img_slice = np.take(img, slice_id, axis=slice_axis)
            X.append(img_slice)
            msk_slice = np.take(msk, slice_id, axis=slice_axis)
            Y.append(msk_slice)

    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def compute_subcubes_count(cube_size, subcube_size, stride):
    """ Compute how many subcubes may be contained inside the original cube """

    num_subcubes = ((cube_size[0] - subcube_size[0]) // stride + 1) * \
                   ((cube_size[1] - subcube_size[1]) // stride + 1) * \
                   ((cube_size[2] - subcube_size[2]) // stride + 1)
    #print(f"Numero sottocubi estraibili: {num_subcubes}")
    return num_subcubes

def extract_subcubes(X, subcube_size, stride):
    """ Apply decomposition to a cube and return subcubes """
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

def reconstruct_dataset_from_subs(X, Y, predictions, cube_shape, subs_shape, stride):
    X_rec = []
    Y_rec = []
    predictions_rec = np.empty(len(predictions), dtype=object)  # ([]*len(predictions))
    print(f"pred shape {predictions_rec.shape}")

    n_subs = compute_subcubes_count(cube_shape, subs_shape, stride)
    print(f"nsubs: {n_subs}")
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

