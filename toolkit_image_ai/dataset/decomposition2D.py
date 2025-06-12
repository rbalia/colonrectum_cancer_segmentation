import matplotlib.pyplot as plt
import numpy as np


""" Collection of functions for managing 2D datasets and performing sliding window decomposition. """

def array2DToTiles(X, Y, tile_size, stride):
    """ Convert a dataset array of fullbubes to an array of subcubes """
    X_tiles = []
    Y_tiles = []
    for x,y in zip(X,Y):
        a = extract_tiles(x, tile_size, stride)
        b = extract_tiles(y, tile_size, stride)

        for im, msk in zip(a, b):
            X_tiles.append(im)
            Y_tiles.append(msk)

    X_tiles = np.array(X_tiles)
    Y_tiles = np.array(Y_tiles)

    #X_tiles = np.array(X_tiles).reshape((-1, *X_tiles.shape[-3:-1], X.shape[-1]))
    #Y_tiles = np.array(Y_tiles).reshape((-1, *X_tiles.shape[-3:-1], X.shape[-1]))
    return X_tiles, Y_tiles

def compute_tiles_count(img_size, tile_size, stride):
    """ Compute how many subcubes may be contained inside the original cube """

    num_tiles = ((img_size[0] - tile_size[0]) // stride + 1) * ((img_size[1] - tile_size[1]) // stride + 1)
    #print(f"Numero sottocubi estraibili: {num_subcubes}")
    return num_tiles

def extract_tiles(X, tile_size, stride):
    """ Apply decomposition to a cube and return subcubes """
    # Calcoliamo il numero totale di sottocubi che possiamo estrarre dall'immagine
    num_tiles = compute_tiles_count(X.shape, tile_size, stride)

    # Inizializziamo un array per contenere i sottocubi estratti
    tiles = np.zeros((num_tiles, *tile_size, X.shape[-1]))

    # Estraiamo i sottocubi utilizzando la finestra scorrevole
    subcube_idx = 0
    for i in range(0, X.shape[0] - tile_size[0] + 1, stride):
        for j in range(0, X.shape[1] - tile_size[1] + 1, stride):
            tile = X[i:i + tile_size[0], j:j + tile_size[1], :]
            tiles[subcube_idx, :, :, :] = tile
            subcube_idx += 1
    return tiles

def reconstruct_dataset_from_tiles(X, Y, predictions, img_shape, tile_shape, stride):
    X_rec = []
    Y_rec = []
    predictions_rec = np.empty(len(predictions), dtype=object)  # ([]*len(predictions))
    print(f"pred shape {predictions_rec.shape}")

    n_subs = compute_tiles_count(img_shape, tile_shape, stride)
    print(f"nsubs: {n_subs}")
    for i in range(0, len(X), n_subs):
        img_reconstruction = reconstruct_image(X[i:i + n_subs], [*img_shape,X.shape[-1]] , tile_shape, stride)
        gt_reconstruction = reconstruct_image(Y[i:i + n_subs], [*img_shape,Y.shape[-1]], tile_shape, stride)
        for j in range(0, len(predictions)):
            if predictions_rec[j] is None:
                predictions_rec[j] = []
            pred_reconstruction = reconstruct_image(predictions[j][i:i + n_subs],
                                                    [*img_shape,Y.shape[-1]],
                                                    tile_shape, stride, mode="avg")
            predictions_rec[j].append(pred_reconstruction)
        X_rec.append(img_reconstruction)
        Y_rec.append(gt_reconstruction)
        # preds_rec.append(pred_reconstruction)
    X_rec = np.array(X_rec)
    Y_rec = np.array(Y_rec)

    # Convert from array of object to array of float
    predictions_rec = np.array(list(predictions_rec), dtype=np.float32)
    #np.array(predictions_rec)
    #np.vstack(predictions_rec[:]).astype(np.float64))

    return X_rec, Y_rec, predictions_rec

def reconstruct_image(subcubes, image_shape, tile_shape, stride, mode="avg"):
    """
    Ricostruisce l'immagine originale a partire dai sottocubi estratti.

    Args:
        subcubes (ndarray): array contenente i sottocubi estratti dall'immagine originale
        image_shape (tuple): tupla contenente le dimensioni dell'immagine originale
        tile_shape (tuple): tupla contenente le dimensioni dei sottocubi
        stride (int): passo di spostamento della finestra scorrevole utilizzata per estrarre i sottocubi

    Returns:
        ndarray: array contenente l'immagine ricostruita
    """
    # Inizializza l'immagine-ricostruita con zeri
    image = np.zeros(image_shape)

    # Inizializza un array che conta il numero di sovrapposizioni di ogni voxel
    overlap_count = np.zeros(image_shape)
    subcube_idx = 0
    # Scorri lungo i sottocubi e aggiungi ogni sottocubo all'immagine ricostruita
    for i in range(0, image_shape[0] - tile_shape[0] + 1, stride):
        for j in range(0, image_shape[1] - tile_shape[1] + 1, stride):
            subcube = subcubes[subcube_idx, :, :, :]
            if mode=="avg":
                #print(f"sub_shape {subcube.shape}")
                #print(f"img_shape {image_shape}")
                image[i:i+tile_shape[0], j:j+tile_shape[1], :] += subcube
                overlap_count[i:i+tile_shape[0], j:j+tile_shape[1], :] += 1
            elif mode=="max":
                image[i:i + tile_shape[0], j:j + tile_shape[1], :] = \
                    np.maximum(image[i:i + tile_shape[0], j:j + tile_shape[1], :],subcube)
            subcube_idx += 1

    # Divide ogni voxel dell'immagine ricostruita per il numero di sovrapposizioni per ottenere la media
    if mode == "avg":
        image /= overlap_count

    return image

