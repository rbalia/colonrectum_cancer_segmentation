import re
import time

import tensorflow as tf
import numpy as np
from keras.layers import Dropout
#import cupy as cp
#import cupyx.scipy
from keras.utils import to_categorical
from skimage.transform import resize
from sklearn.utils import shuffle

from toolkit_image_ai.processing.augmentation2D import dataAugmentationGenerator
from toolkit_image_ai.processing.augmentation3D import image_augmentation
from toolkit_image_ai.dataset.decomposition2D import extract_tiles, compute_tiles_count
from toolkit_image_ai.plot.text_formatting import TxtClr as tc
#from src_analisys_modules.preprocessing import filter_empty_cubes, split_valid_empty, count_ditribution

def find_test_batch_size(num, llim=2):
    # Num is the "tiles per image"
    for i in range(llim, num):
        if num % i == 0:
            return num // i

def find_train_batch_size(num, ulim, llim=2):
    # Num is the "buffer size"
    # ulim è il batch size corrente che ha alzato un eventuale errore nei controlli del datagenerator
    # Cerca un batch size inferiore che divida il buffer
    for i in np.flip(np.arange(llim, ulim)):
        #print(f"i:{i} - num:{num} mod:{num%i}")
        if num % i == 0:
            return i
    # if no solution found
    return ulim

        # pensato per il training, se non
class DataGeneratorSAT(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    # TODO: Introduce Support to:
    #   X, Y come array invece che come path
    #
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, tile_shape=(128,128),  n_classes=20,
                 n_img_channels=3, n_msk_channels=20, shuffle=True, image_shape=(1920, 960), stride=32, n_img=4, n_aug=5,
                 dtype=np.float16, list_type="path", apply_augmentation=True, apply_balancing=False,
                 allow_partilly_filled_batch=False, force_tensor_conversion=True, return_weights=False, weights=None):

        print("")
        """ Initialization """
        # Preprocessing params
        self.image_shape = image_shape
        self.tile_shape = tile_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing
        self.dtype = dtype
        self.weights = weights

        # File Loading params
        self.n_img = n_img
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list
        self.list_type = list_type

        # Model params
        self.batch_size = batch_size
        self.n_img_channels = n_img_channels
        self.n_msk_channels = n_msk_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.allow_partilly_filled_batch = allow_partilly_filled_batch
        self.force_tensor_conversion = force_tensor_conversion
        self.return_weights = return_weights
        self.tiles_per_image = compute_tiles_count(image_shape, tile_shape, stride)
        self.tiles_per_buffer = self.tiles_per_image * self.n_img
        self.X_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_img_channels), dtype=self.dtype)
        self.Y_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_msk_channels), dtype=self.dtype)
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        if self.allow_partilly_filled_batch:
            # Include partially filled batch
            return int(np.ceil((len(self.list_IDs)*self.tiles_per_image*self.n_aug ) / self.batch_size))
        else:
            # Exclude partially filled batch (Default)
            return int(np.floor((len(self.list_IDs) * self.tiles_per_image * self.n_aug) / self.batch_size))



    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = (batch_id) % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_img:(self.cube_batch_id + 1) * self.n_img]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_img > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        if self.force_tensor_conversion:
            with tf.device('/cpu:0'):
                if tf.is_tensor(X) == False:
                    X = tf.convert_to_tensor(X, self.dtype)
                    Y = tf.convert_to_tensor(Y, self.dtype)

        if self.weights is None:
            self.weights = [1]*self.n_classes

        #print(f"$DG: {self.weights}")
        if self.return_weights:
            return X, Y, self.weights
        else:
            return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    def __fill_buffer(self, list_IDs_temp):

        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")

        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            lowlim = i*self.tiles_per_image
            uprlim = (i+1)*self.tiles_per_image

            # Load sample
            if self.list_type == "path":
                #print(f"{ID} - {self.X_path_list[ID]}")
                x = np.load(self.X_path_list[ID], mmap_mode='r')
                #x = resize(x, (self.image_shape), mode='constant')
                #x = x / np.max(x) # Normalization (already done in preprocessing)
            elif self.list_type == "array":
                x = self.X_path_list[ID]

            # Load ground truth
            if self.list_type == "path":
                y = np.load(self.Y_path_list[ID], mmap_mode='r')
                #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            elif self.list_type == "array":
                y = self.Y_path_list[ID]

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = dataAugmentationGenerator(x, y, 1-(1/self.n_aug), out_dtype=self.dtype)

            # Extract subcubes and fill buffer
            if self.tiles_per_image > 1:
                self.X_buffer[lowlim:uprlim] = extract_tiles(x, self.tile_shape, self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_tiles(y, self.tile_shape, self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.Y_buffer[lowlim:uprlim] = y

        # Shuffle
        if self.shuffle:
           self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        """if self.apply_balancing:
            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer),
                                                                   verbose=False)
            while len(X_valid) < len(self.X_buffer)//2:
                #print(f"Buffer filled with {len(X_valid)}/{len(self.X_buffer)//2}")
                lm = (len(self.X_buffer)//2) - len(X_valid)
                if lm > len(X_valid):
                    lm = len(X_valid)
                #print(f"Adding {lm} subcubes")
                for i in range(0,lm):
                    x, y = image_augmentation(X_valid[i], Y_valid[i], 0.7)
                    if np.count_nonzero(y)>10:
                        X_valid.append(x)
                        Y_valid.append(y)

            v_id = 0
            e_id = 0
            self.X_buffer *= 0
            self.Y_buffer *= 0
            for i in range(0, len(self.X_buffer)):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1"""

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

    def get_dataset(self, apply_decomposition, apply_flattening):
        X = []
        Y = []
        for i, ID in enumerate(self.list_IDs):
            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            #x = resize(x, (self.image_shape), mode='constant')
            #x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_tiles(x, (*self.tile_shape,), self.stride)
                y = extract_tiles(y, (*self.tile_shape,), self.stride)
            if apply_flattening and apply_decomposition:
                for x_sub, y_sub in zip(x,y):
                    X.append(x_sub)
                    Y.append(y_sub)
            else:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y

class DataGeneratorSATV2(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria

    Novità V2:
        -- Aggiunti controlli per segnalare la perdita di informazioni (tiles che non vengono mai restituite)
        -- Aggiunte meccaniche per restituire batch non completamente pieni: L'ultimo batch potrebbe contenere meno
            tiles in quanto si restituiscono elementi del buffer sino all'indice a cui è stato riempito, sono evitati
            quindi troncamenti a buffer non completamente pieni o a buffer con tiles nere, a patto di indicare la
            politica di restituzione (allow_partilly_filled_batch (ora true di default))
    """

    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, tile_shape=(128,128),  n_classes=20,
                 n_img_channels=3, n_msk_channels=20, shuffle=True, image_shape=(1920, 960), stride=32, n_img=4, n_aug=5,
                 aug_prob=None, dtype=np.float16, list_type="path", apply_augmentation=True, apply_balancing=False,
                 allow_partilly_filled_batch=False, force_tensor_conversion=True, return_weights=False, weights=None,
                 name="Generic DataGeneratorV2", step_msg=True, error_msg=True, enable_resize=False):

        self.name = name
        if step_msg:
            print(f"\n{tc.green}Datagenerator '{self.name}'{tc.end}: Initialization")

        """ Initialization """
        # Preprocessing params
        self.image_shape = image_shape
        self.tile_shape = tile_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        if aug_prob is None:
            self.aug_prob = 1-(1/self.n_aug)
        else:
            self.aug_prob = aug_prob
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing
        self.dtype = dtype
        self.weights = weights

        # File Loading params
        self.n_img = n_img
        self.list_IDs = list_IDs
        self.list_IDs_repeated = np.repeat([list_IDs], n_aug, axis=0).ravel()
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list
        self.list_type = list_type
        self.enable_resize = enable_resize

        # Model params
        self.batch_size = batch_size
        self.n_img_channels = n_img_channels
        self.n_msk_channels = n_msk_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.allow_partilly_filled_batch = allow_partilly_filled_batch
        self.force_tensor_conversion = force_tensor_conversion
        self.return_weights = return_weights
        self.tiles_per_image = compute_tiles_count(image_shape, tile_shape, stride)
        self.tiles_per_buffer = self.tiles_per_image * self.n_img
        #print(f"tiles per buffer = { self.tiles_per_buffer}")
        #print(f"augs: {self.n_aug}")
        self.X_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_img_channels), dtype=self.dtype)
        self.Y_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_msk_channels), dtype=self.dtype)

        # FOR DEBUGGING
        self.X_path_buffer = []
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1

        """ On Development """
        self.buffer_last_index = len(self.X_buffer)
        self.batch_id_bypass = -1
        self.batch_anomaly_detected = False

        """ Controlli """
        if int(np.ceil((len(self.X_buffer) / self.batch_size))) > (len(self.X_buffer) // self.batch_size):
            print(f"\t{tc.red}DatagenError{tc.end}: information loss detected: Buffer not divisible by batch")
            self.batch_size = find_train_batch_size(len(self.X_buffer), self.batch_size)
            print(f"\t\tBatch size forcefully changed to {self.batch_size}")
            #time.sleep(10)

        if not 1 >= self.aug_prob >= 0:
            print(f"\t{tc.red}DatagenError{tc.end}: augmentation probability out of range")
            self.aug_prob = np.clip(self.aug_prob, 0.0, 1.0)
            print(f"\t\tAugmentation probability clipped to {self.aug_prob}")
            #time.sleep(10)


        self.on_epoch_end()
        if step_msg:
            print(f"{tc.green}Datagenerator '{self.name}'{tc.end}: Built. \n\t -- Remember to set model.fit(shuffle=False) ;)")

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        if self.allow_partilly_filled_batch:
            # Include partially filled batch
            #print(f"Allow partially {int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image ) / self.batch_size))}")
            return int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image ) / self.batch_size))
        else:
            # Exclude partially filled batch (Default)
            #print(f"Exclude {int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / self.batch_size))}")
            return int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / self.batch_size))



    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        #self.prev_batch_id = self.batch_id_bypass

        #if batch_id == 0 and self.prev_batch_id<=0:
        #    self.batch_id_bypass = 0
        #else:
        #    self.batch_id_bypass += 1

        # Se si rilava un problema che non era stato rilevato prima:
        if self.batch_anomaly_detected == False:
            if abs(batch_id - self.prev_batch_id) > 1:
                print(f"\n{tc.red}DatagenError{tc.end}: DataGenerator-Unrelated batch shuffle detected.")
                print(f"{tc.red}DatagenError{tc.end}: Set 'shuffle=False' in model.fit to suppress this error and unexpected behaviors")
                print(f"{tc.red}DatagenError{tc.end}: Batch ID bypass system enabled")
                self.batch_anomaly_detected = True
                time.sleep(10)
        # Se è stata rilevata una anomalia
        if self.batch_anomaly_detected == True:
            if self.prev_batch_id == 0 and abs(self.prev_batch_id-self.prev_batch_id_bypass)<=1:
                self.batch_id_bypass = 0
            else:
                self.batch_id_bypass += 1
        # Se nessun problema è rilevato:
        else:
            self.batch_id_bypass = batch_id

        #print(f"\nReceived a call with batch id: {batch_id}, using {self.batch_id_bypass}, previus: {self.prev_batch_id}")
        # Edit the batch ID in order to cyclicaly read from buffer
        # TODO:
        #  [1] Se la decomposizione è attiva è possibile che vengano scartati batch non pienamente riempiti.
        #       in fase di test questo causa il recupero incompleto dei campioni
        #  [2] Se lo shuffle su model.fit non è esplicitamente settato a False, il calcolo di batch_id_ fallisce
        batch_per_buffer = int(np.floor(len(self.X_buffer) / self.batch_size))
        batch_id_ = (self.batch_id_bypass) % batch_per_buffer

        # Reset the cube_batch_id if tensorflow called a tracing operation for building the graph
        if self.batch_id_bypass == 0:
            self.cube_batch_id = 0

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed and the same batch has not requested in previous bath (due to tracing)
        if batch_id_ == 0:

            #if self.batch_id_bypass == self.prev_batch_id_bypass:
            #    print(f"{tc.yellow}DatagenWarning{tc.end}: Detected an attempt to refill the Buffer with the same data")


            #print(f"\nbatch_id_ : {batch_id_} - batch_id : {batch_id} - cuba_batch_id: {self.cube_batch_id}")
            #time.sleep(3)

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs_repeated[self.cube_batch_id * self.n_img:
                                                   (self.cube_batch_id + 1) * self.n_img]

            # Fill the buffer
            #if self.batch_id_bypass != self.prev_batch_id_bypass:
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1


            # If the cubes have been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id) * self.n_img > len(self.list_IDs_repeated):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            # TODO: In origine gli indici erano sulla lunghezza del buffer (len(X_buffer), con self.buffer_last_index
            #  si tiene in considerazione di quanto il buffer è stato realmente riempito pertanto può restituire un
            #  ULTIMA batch più piccola
            self.subs_cube_indexes = np.arange(self.buffer_last_index)

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]
        #print(f"\nreturning a batch of size {len(subs_indexes)}")
        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        if self.force_tensor_conversion:
            with tf.device('/cpu:0'):
                if tf.is_tensor(X) == False:
                    X = tf.convert_to_tensor(X, self.dtype)
                    Y = tf.convert_to_tensor(Y, self.dtype)

        if self.weights is None:
            """total_pixel_count = np.sum(Y) * 0.01
            label_pixel_count_sum = np.sum(Y, axis=(0,1,2)) / total_pixel_count
            _weights = np.reshape(label_pixel_count_sum, (1, self.n_msk_channels))  # /100
            _weights = (100 - _weights) / 100
            self.weights = _weights"""
            self.weights = [1]*self.n_msk_channels

        # Prima di ritornare, salva il numero di questa batch
        self.prev_batch_id = batch_id
        self.prev_batch_id_bypass = self.batch_id_bypass

        #print(f"$DG: {self.weights}")
        if self.return_weights:
            return X, Y, self.weights
        else:
            #print("returning data")
            #mask = Dropout(0.1)(np.ones_like(X[...,0:1]), training=True)
            #X = np.array(X)*np.array(mask)
            return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

        # Reset pivotting indexes
        self.cube_batch_id = 0
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1
        self.batch_id_bypass = -1
        #self.buffer_last_index = len(self.X_buffer)

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs_repeated)

    def __fill_buffer(self, list_IDs_temp):

        #print(f"\r{tc.green}Datagenerator '{self.name}'{tc.end}: Filling the buffer with {len(list_IDs_temp)} new samples", end="")

        self.X_path_buffer = []
        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")
        self.buffer_last_index = 0
        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            lowlim = i*self.tiles_per_image
            uprlim = (i+1)*self.tiles_per_image

            # Load sample
            if self.list_type == "path":
                x = np.load(self.X_path_list[ID], mmap_mode="r")
                if self.enable_resize:
                    x = resize(x, (self.image_shape), mode='constant')
                #x = x / np.max(x) # Normalization (already done in preprocessing)
            elif self.list_type == "array":
                x = self.X_path_list[ID]

            # Load ground truth
            if self.list_type == "path":
                #print(f"v2_{ID} - {self.X_path_list[ID]}", end="\r")
                y = np.load(self.Y_path_list[ID], mmap_mode="r")
                if self.enable_resize:
                    y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            elif self.list_type == "array":
                y = self.Y_path_list[ID]

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = dataAugmentationGenerator(x, y, self.aug_prob, out_dtype=self.dtype)

            # Extract subcubes and fill buffer
            if self.tiles_per_image > 1:
                self.X_buffer[lowlim:uprlim] = extract_tiles(x, self.tile_shape, self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_tiles(y, self.tile_shape, self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.X_path_buffer.append(self.X_path_list[ID])
                self.Y_buffer[lowlim:uprlim] = y

            self.buffer_last_index = uprlim

        # Shuffle
        if self.shuffle:
           (self.X_buffer[0:self.buffer_last_index],
            self.Y_buffer[0:self.buffer_last_index]) = (
               shuffle(self.X_buffer[0:self.buffer_last_index],
                       self.Y_buffer[0:self.buffer_last_index]))

        """if self.apply_balancing:
            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer),
                                                                   verbose=False)
            while len(X_valid) < len(self.X_buffer)//2:
                #print(f"Buffer filled with {len(X_valid)}/{len(self.X_buffer)//2}")
                lm = (len(self.X_buffer)//2) - len(X_valid)
                if lm > len(X_valid):
                    lm = len(X_valid)
                #print(f"Adding {lm} subcubes")
                for i in range(0,lm):
                    x, y = image_augmentation(X_valid[i], Y_valid[i], 0.7)
                    if np.count_nonzero(y)>10:
                        X_valid.append(x)
                        Y_valid.append(y)

            v_id = 0
            e_id = 0
            self.X_buffer *= 0
            self.Y_buffer *= 0
            for i in range(0, len(self.X_buffer)):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1"""

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """
        #for i in subs_indexes:
        #    print(self.X_path_buffer[i] end="\r")
        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

    def get_dataset(self, apply_decomposition, apply_flattening, range=None):
        X = []
        Y = []
        list_IDs = np.copy(self.list_IDs)
        if range is not None:
            list_IDs = list_IDs[range[0]:range[1]]
        for i, ID in enumerate(list_IDs):
            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            if self.enable_resize:
                x = resize(x, (self.image_shape), mode='constant')
            #x = resize(x, (self.image_shape), mode='constant')
            #x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            if self.enable_resize:
                y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_tiles(x, (*self.tile_shape,), self.stride)
                y = extract_tiles(y, (*self.tile_shape,), self.stride)
            if apply_flattening and apply_decomposition:
                for x_sub, y_sub in zip(x,y):
                    X.append(x_sub)
                    Y.append(y_sub)
            else:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y


class DataGeneratorSATV2_mixedTiles(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria

    Novità V2:
        -- Aggiunti controlli per segnalare la perdita di informazioni (tiles che non vengono mai restituite)
        -- Aggiunte meccaniche per restituire batch non completamente pieni: L'ultimo batch potrebbe contenere meno
            tiles in quanto si restituiscono elementi del buffer sino all'indice a cui è stato riempito, sono evitati
            quindi troncamenti a buffer non completamente pieni o a buffer con tiles nere, a patto di indicare la
            politica di restituzione (allow_partilly_filled_batch (ora true di default))
    """

    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, tile_shape=(128, 128), n_classes=20,
                 n_img_channels=3, n_msk_channels=20, shuffle=True, image_shape=(1920, 960), stride=32, n_img=4,
                 n_aug=5,
                 aug_prob=None, dtype=np.float16, list_type="path", apply_augmentation=True, apply_balancing=False,
                 allow_partilly_filled_batch=False, force_tensor_conversion=True, return_weights=False, weights=None,
                 name="Generic DataGeneratorV2", step_msg=True, error_msg=True):

        self.name = name
        if step_msg:
            print(f"\n{tc.green}Datagenerator '{self.name}'{tc.end}: Initialization")

        """ Initialization """
        # Preprocessing params
        self.image_shape = image_shape
        self.tile_shape = tile_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        if aug_prob is None:
            self.aug_prob = 1 - (1 / self.n_aug)
        else:
            self.aug_prob = aug_prob
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing
        self.dtype = dtype
        self.weights = weights

        # File Loading params
        self.n_img = n_img
        self.list_IDs = list_IDs
        self.list_IDs_repeated = np.repeat([list_IDs], n_aug, axis=0).ravel()
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list
        self.list_type = list_type

        # Model params
        self.batch_size = batch_size
        self.n_img_channels = n_img_channels
        self.n_msk_channels = n_msk_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.allow_partilly_filled_batch = allow_partilly_filled_batch
        self.force_tensor_conversion = force_tensor_conversion
        self.return_weights = return_weights
        self.tiles_per_image = compute_tiles_count(image_shape, tile_shape, stride)
        self.tiles_per_buffer = self.tiles_per_image * self.n_img
        # print(f"tiles per buffer = { self.tiles_per_buffer}")
        # print(f"augs: {self.n_aug}")
        self.X_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_img_channels), dtype=self.dtype)
        self.Y_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_msk_channels), dtype=self.dtype)

        # FOR DEBUGGING
        self.X_path_buffer = []
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1

        """ On Development """
        self.buffer_last_index = len(self.X_buffer)
        self.batch_id_bypass = -1
        self.batch_anomaly_detected = False

        """ Controlli """
        if int(np.ceil((len(self.X_buffer) / self.batch_size))) > (len(self.X_buffer) // self.batch_size):
            print(f"\t{tc.red}DatagenError{tc.end}: information loss detected: Buffer not divisible by batch")
            self.batch_size = find_train_batch_size(len(self.X_buffer), self.batch_size)
            print(f"\t\tBatch size forcefully changed to {self.batch_size}")
            # time.sleep(10)

        if not 1 >= self.aug_prob >= 0:
            print(f"\t{tc.red}DatagenError{tc.end}: augmentation probability out of range")
            self.aug_prob = np.clip(self.aug_prob, 0.0, 1.0)
            print(f"\t\tAugmentation probability clipped to {self.aug_prob}")
            # time.sleep(10)

        self.on_epoch_end()
        if step_msg:
            print(
                f"{tc.green}Datagenerator '{self.name}'{tc.end}: Built. \n\t -- Remember to set model.fit(shuffle=False) ;)")

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        if self.allow_partilly_filled_batch:
            # Include partially filled batch
            # print(f"Allow partially {int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image ) / self.batch_size))}")
            return int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image) / (self.batch_size)))//4
        else:
            # Exclude partially filled batch (Default)
            # print(f"Exclude {int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / self.batch_size))}")
            return int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / (self.batch_size)))//4

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # self.prev_batch_id = self.batch_id_bypass

        # if batch_id == 0 and self.prev_batch_id<=0:
        #    self.batch_id_bypass = 0
        # else:
        #    self.batch_id_bypass += 1

        # Se si rilava un problema che non era stato rilevato prima:
        if self.batch_anomaly_detected == False:
            if abs(batch_id - self.prev_batch_id) > 1:
                print(f"\n{tc.red}DatagenError{tc.end}: DataGenerator-Unrelated batch shuffle detected.")
                print(
                    f"{tc.red}DatagenError{tc.end}: Set 'shuffle=False' in model.fit to suppress this error and unexpected behaviors")
                print(f"{tc.red}DatagenError{tc.end}: Batch ID bypass system enabled")
                self.batch_anomaly_detected = True
                time.sleep(10)
        # Se è stata rilevata una anomalia
        if self.batch_anomaly_detected == True:
            if self.prev_batch_id == 0 and abs(self.prev_batch_id - self.prev_batch_id_bypass) <= 1:
                self.batch_id_bypass = 0
            else:
                self.batch_id_bypass += 1
        # Se nessun problema è rilevato:
        else:
            self.batch_id_bypass = batch_id

        # print(f"\nReceived a call with batch id: {batch_id}, using {self.batch_id_bypass}, previus: {self.prev_batch_id}")
        # Edit the batch ID in order to cyclicaly read from buffer
        # TODO:
        #  [1] Se la decomposizione è attiva è possibile che vengano scartati batch non pienamente riempiti.
        #       in fase di test questo causa il recupero incompleto dei campioni
        #  [2] Se lo shuffle su model.fit non è esplicitamente settato a False, il calcolo di batch_id_ fallisce
        batch_per_buffer = int(np.floor(len(self.X_buffer) / self.batch_size))
        batch_id_ = (self.batch_id_bypass) % batch_per_buffer

        # Reset the cube_batch_id if tensorflow called a tracing operation for building the graph
        if self.batch_id_bypass == 0:
            self.cube_batch_id = 0

        # print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed and the same batch has not requested in previous bath (due to tracing)
        if batch_id_ == 0:

            # if self.batch_id_bypass == self.prev_batch_id_bypass:
            #    print(f"{tc.yellow}DatagenWarning{tc.end}: Detected an attempt to refill the Buffer with the same data")

            # print(f"\nbatch_id_ : {batch_id_} - batch_id : {batch_id} - cuba_batch_id: {self.cube_batch_id}")
            # time.sleep(3)

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs_repeated[self.cube_batch_id * self.n_img:
                                                   (self.cube_batch_id + 1) * self.n_img]

            # Fill the buffer
            # if self.batch_id_bypass != self.prev_batch_id_bypass:
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # If the cubes have been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id) * self.n_img > len(self.list_IDs_repeated):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            # TODO: In origine gli indici erano sulla lunghezza del buffer (len(X_buffer), con self.buffer_last_index
            #  si tiene in considerazione di quanto il buffer è stato realmente riempito pertanto può restituire un
            #  ULTIMA batch più piccola
            self.subs_cube_indexes = np.arange(self.buffer_last_index)

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]
        # print(f"\nreturning a batch of size {len(subs_indexes)}")
        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        if self.force_tensor_conversion:
            with tf.device('/cpu:0'):
                if tf.is_tensor(X) == False:
                    X = tf.convert_to_tensor(X, self.dtype)
                    Y = tf.convert_to_tensor(Y, self.dtype)

        if self.weights is None:
            """total_pixel_count = np.sum(Y) * 0.01
            label_pixel_count_sum = np.sum(Y, axis=(0,1,2)) / total_pixel_count
            _weights = np.reshape(label_pixel_count_sum, (1, self.n_msk_channels))  # /100
            _weights = (100 - _weights) / 100
            self.weights = _weights"""
            self.weights = [1] * self.n_msk_channels

        # Prima di ritornare, salva il numero di questa batch
        self.prev_batch_id = batch_id
        self.prev_batch_id_bypass = self.batch_id_bypass

        # print(f"$DG: {self.weights}")
        if self.return_weights:
            return X, Y, self.weights
        else:
            # print("returning data")
            # mask = Dropout(0.1)(np.ones_like(X[...,0:1]), training=True)
            # X = np.array(X)*np.array(mask)
            return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

        # Reset pivotting indexes
        self.cube_batch_id = 0
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1
        self.batch_id_bypass = -1
        # self.buffer_last_index = len(self.X_buffer)

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs_repeated)

    def __fill_buffer(self, list_IDs_temp):

        # print(f"\r{tc.green}Datagenerator '{self.name}'{tc.end}: Filling the buffer with {len(list_IDs_temp)} new samples", end="")

        self.X_path_buffer = []
        # print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")
        self.buffer_last_index = 0
        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            lowlim = i * self.tiles_per_image
            uprlim = (i + 1) * self.tiles_per_image

            # Load sample
            if self.list_type == "path":
                x = np.load(self.X_path_list[ID], mmap_mode="r")
                # x = resize(x, (self.image_shape), mode='constant')
                # x = x / np.max(x) # Normalization (already done in preprocessing)
            elif self.list_type == "array":
                x = self.X_path_list[ID]

            # Load ground truth
            if self.list_type == "path":
                # print(f"v2_{ID} - {self.X_path_list[ID]}", end="\r")
                y = np.load(self.Y_path_list[ID], mmap_mode="r")
                # y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            elif self.list_type == "array":
                y = self.Y_path_list[ID]

            # Apply random augmentation
            if self.apply_augmentation:
                x, y = dataAugmentationGenerator(x, y, self.aug_prob, out_dtype=self.dtype)

            # Extract subcubes and fill buffer
            if self.tiles_per_image > 1:
                self.X_buffer[lowlim:uprlim] = extract_tiles(x, self.tile_shape, self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_tiles(y, self.tile_shape, self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.X_path_buffer.append(self.X_path_list[ID])
                self.Y_buffer[lowlim:uprlim] = y

            self.buffer_last_index = uprlim

        # Shuffle
        if self.shuffle:
            (self.X_buffer[0:self.buffer_last_index],
             self.Y_buffer[0:self.buffer_last_index]) = (
                shuffle(self.X_buffer[0:self.buffer_last_index],
                        self.Y_buffer[0:self.buffer_last_index]))

        """if self.apply_balancing:
            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer),
                                                                   verbose=False)
            while len(X_valid) < len(self.X_buffer)//2:
                #print(f"Buffer filled with {len(X_valid)}/{len(self.X_buffer)//2}")
                lm = (len(self.X_buffer)//2) - len(X_valid)
                if lm > len(X_valid):
                    lm = len(X_valid)
                #print(f"Adding {lm} subcubes")
                for i in range(0,lm):
                    x, y = image_augmentation(X_valid[i], Y_valid[i], 0.7)
                    if np.count_nonzero(y)>10:
                        X_valid.append(x)
                        Y_valid.append(y)

            v_id = 0
            e_id = 0
            self.X_buffer *= 0
            self.Y_buffer *= 0
            for i in range(0, len(self.X_buffer)):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1"""

        ## Shuffle
        # if self.shuffle:
        #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

    def __data_generation(self, subs_indexes):
        """ Load a batch of data from buffer """
        # for i in subs_indexes:
        #    print(self.X_path_buffer[i] end="\r")
        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]
        tile_shape_ = self.tile_shape[0]

        Xmix = np.zeros((self.batch_size//4, tile_shape_*2,tile_shape_*2, self.n_img_channels), dtype=self.dtype)
        Ymix = np.zeros((self.batch_size//4, tile_shape_*2,tile_shape_*2, self.n_msk_channels), dtype=self.dtype)

        #print(X.shape)
        #print(X[(self.batch_size//4)*0:(self.batch_size//4)*1,...].shape)
        #print(Xmix.shape)
        Xmix[:,0:tile_shape_,0:tile_shape_,:] = X[(self.batch_size//4)*0:(self.batch_size//4)*1,...]
        Ymix[:,0:tile_shape_,0:tile_shape_,:] = Y[(self.batch_size//4)*0:(self.batch_size//4)*1,...]

        Xmix[:, 0:tile_shape_, tile_shape_:, :] = X[self.batch_size // 4 * 1:self.batch_size // 4 * 2,...]
        Ymix[:, 0:tile_shape_, tile_shape_:, :] = Y[self.batch_size // 4 * 1:self.batch_size // 4 * 2,...]

        Xmix[:, tile_shape_:, 0:tile_shape_, :] = X[self.batch_size // 4 * 2:self.batch_size // 4 * 3]
        Ymix[:, tile_shape_:, 0:tile_shape_, :] = Y[self.batch_size // 4 * 2:self.batch_size // 4 * 3]

        Xmix[:, tile_shape_:, tile_shape_:, :] = X[self.batch_size // 4 * 3:]
        Ymix[:, tile_shape_:, tile_shape_:, :] = Y[self.batch_size // 4 * 3:]

        return Xmix, Ymix

    def get_dataset(self, apply_decomposition, apply_flattening, range=None):
        X = []
        Y = []
        list_IDs = np.copy(self.list_IDs)
        if range is not None:
            list_IDs = list_IDs[range[0]:range[1]]
        for i, ID in enumerate(list_IDs):
            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            # x = resize(x, (self.image_shape), mode='constant')
            # x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            # y = resize(y, (self.image_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_tiles(x, (*self.tile_shape,), self.stride)
                y = extract_tiles(y, (*self.tile_shape,), self.stride)
            if apply_flattening and apply_decomposition:
                for x_sub, y_sub in zip(x, y):
                    X.append(x_sub)
                    Y.append(y_sub)
            else:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X, Y



class DataGeneratorSATV2_dualModal(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria

    Novità V2:
        -- Aggiunti controlli per segnalare la perdita di informazioni (tiles che non vengono mai restituite)
        -- Aggiunte meccaniche per restituire batch non completamente pieni: L'ultimo batch potrebbe contenere meno
            tiles in quanto si restituiscono elementi del buffer sino all'indice a cui è stato riempito, sono evitati
            quindi troncamenti a buffer non completamente pieni o a buffer con tiles nere, a patto di indicare la
            politica di restituzione (allow_partilly_filled_batch (ora true di default))
    """

    def __init__(self, list_IDs, X1_path_list, X2_path_list, Y_path_list, batch_size=32, tile_shape=(128,128), n_classes=20,
                 x1_channels=3, x2_channels=3, n_msk_channels=20, shuffle=True, image_shape=(1920, 960), stride=32, n_img=4, n_aug=5,
                 aug_prob=None, dtype=np.float16, list_type="path", apply_augmentation=True, apply_balancing=False,
                 allow_partilly_filled_batch=False, force_tensor_conversion=True, return_weights=False, weights=None,
                 name="Generic DataGeneratorV2", concat_output=False):

        self.name = name
        print(f"\n{tc.green}Datagenerator '{self.name}'{tc.end}: Initialization")

        """ Initialization """
        # Preprocessing params
        self.image_shape = image_shape
        self.tile_shape = tile_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        if aug_prob is None:
            self.aug_prob = 1-(1/self.n_aug)
        else:
            self.aug_prob = aug_prob
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing
        self.dtype = dtype
        self.weights = weights

        # File Loading params
        self.n_img = n_img
        self.list_IDs = list_IDs
        self.list_IDs_repeated = np.repeat([list_IDs], n_aug, axis=0).ravel()
        self.X1_path_list = X1_path_list
        self.X2_path_list = X2_path_list
        self.Y_path_list = Y_path_list
        self.list_type = list_type

        # Model params
        self.batch_size = batch_size
        self.x1_channels = x1_channels
        self.x2_channels = x2_channels
        self.n_msk_channels = n_msk_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.allow_partilly_filled_batch = allow_partilly_filled_batch
        self.force_tensor_conversion = force_tensor_conversion
        self.return_weights = return_weights
        self.tiles_per_image = compute_tiles_count(image_shape, tile_shape, stride)
        self.tiles_per_buffer = self.tiles_per_image * self.n_img
        #print(f"tiles per buffer = { self.tiles_per_buffer}")
        #print(f"augs: {self.n_aug}")
        self.X1_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.x1_channels), dtype=self.dtype)
        self.X2_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.x2_channels), dtype=self.dtype)
        self.Y_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_msk_channels), dtype=self.dtype)

        # FOR DEBUGGING
        self.X_path_buffer = []
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1

        """ On Development """
        self.buffer_last_index = len(self.X1_buffer)
        self.batch_id_bypass = -1
        self.batch_anomaly_detected = False
        self.concat_output = concat_output

        """ Controlli """
        if int(np.ceil((len(self.X1_buffer) / self.batch_size))) > (len(self.X1_buffer) // self.batch_size):
            print(f"\t{tc.red}DatagenError{tc.end}: information loss detected: Buffer not divisible by batch")
            self.batch_size = find_train_batch_size(len(self.X1_buffer), self.batch_size)
            print(f"\t\tBatch size forcefully changed to {self.batch_size}")
            #time.sleep(10)

        if not 1 >= self.aug_prob >= 0:
            print(f"\t{tc.red}DatagenError{tc.end}: augmentation probability out of range")
            self.aug_prob = np.clip(self.aug_prob, 0.0, 1.0)
            print(f"\t\tAugmentation probability clipped to {self.aug_prob}")
            #time.sleep(10)


        self.on_epoch_end()
        print(f"{tc.green}Datagenerator '{self.name}'{tc.end}: Built. \n\t -- Remember to set model.fit(shuffle=False) ;)")

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        if self.allow_partilly_filled_batch:
            # Include partially filled batch
            #print(f"Allow partially {int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image ) / self.batch_size))}")
            return int(np.ceil((len(self.list_IDs_repeated) * self.tiles_per_image ) / self.batch_size))
        else:
            # Exclude partially filled batch (Default)
            #print(f"Exclude {int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / self.batch_size))}")
            return int(np.floor((len(self.list_IDs_repeated) * self.tiles_per_image) / self.batch_size))



    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        #self.prev_batch_id = self.batch_id_bypass

        #if batch_id == 0 and self.prev_batch_id<=0:
        #    self.batch_id_bypass = 0
        #else:
        #    self.batch_id_bypass += 1

        # Se si rilava un problema che non era stato rilevato prima:
        if self.batch_anomaly_detected == False:
            if abs(batch_id - self.prev_batch_id) > 1:
                print(f"\n{tc.red}DatagenError{tc.end}: DataGenerator-Unrelated batch shuffle detected.")
                print(f"{tc.red}DatagenError{tc.end}: Set 'shuffle=False' in model.fit to suppress this error and unexpected behaviors")
                print(f"{tc.red}DatagenError{tc.end}: Batch ID bypass system enabled")
                self.batch_anomaly_detected = True
                time.sleep(10)
        # Se è stata rilevata una anomalia
        if self.batch_anomaly_detected == True:
            if self.prev_batch_id == 0 and abs(self.prev_batch_id-self.prev_batch_id_bypass)<=1:
                self.batch_id_bypass = 0
            else:
                self.batch_id_bypass += 1
        # Se nessun problema è rilevato:
        else:
            self.batch_id_bypass = batch_id

        #print(f"\nReceived a call with batch id: {batch_id}, using {self.batch_id_bypass}, previus: {self.prev_batch_id}")
        # Edit the batch ID in order to cyclicaly read from buffer
        # TODO:
        #  [1] Se la decomposizione è attiva è possibile che vengano scartati batch non pienamente riempiti.
        #       in fase di test questo causa il recupero incompleto dei campioni
        #  [2] Se lo shuffle su model.fit non è esplicitamente settato a False, il calcolo di batch_id_ fallisce
        batch_per_buffer = int(np.floor(len(self.X1_buffer) / self.batch_size))
        batch_id_ = (self.batch_id_bypass) % batch_per_buffer

        # Reset the cube_batch_id if tensorflow called a tracing operation for building the graph
        if self.batch_id_bypass == 0:
            self.cube_batch_id = 0

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed and the same batch has not requested in previous bath (due to tracing)
        if batch_id_ == 0:

            #if self.batch_id_bypass == self.prev_batch_id_bypass:
            #    print(f"{tc.yellow}DatagenWarning{tc.end}: Detected an attempt to refill the Buffer with the same data")


            #print(f"\nbatch_id_ : {batch_id_} - batch_id : {batch_id} - cuba_batch_id: {self.cube_batch_id}")
            #time.sleep(3)

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs_repeated[self.cube_batch_id * self.n_img:
                                                   (self.cube_batch_id + 1) * self.n_img]

            # Fill the buffer
            #if self.batch_id_bypass != self.prev_batch_id_bypass:
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1


            # If the cubes have been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id) * self.n_img > len(self.list_IDs_repeated):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            # TODO: In origine gli indici erano sulla lunghezza del buffer (len(X_buffer), con self.buffer_last_index
            #  si tiene in considerazione di quanto il buffer è stato realmente riempito pertanto può restituire un
            #  ULTIMA batch più piccola
            self.subs_cube_indexes = np.arange(self.buffer_last_index)

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]
        #print(f"\nreturning a batch of size {len(subs_indexes)}")
        # Load a batch of data from buffer
        X1, X2, Y = self.__data_generation(subs_indexes)

        if self.force_tensor_conversion:
            with tf.device('/cpu:0'):
                if tf.is_tensor(Y) == False:
                    X1 = tf.convert_to_tensor(X1, self.dtype)
                    X2 = tf.convert_to_tensor(X2, self.dtype)
                    Y = tf.convert_to_tensor(Y, self.dtype)

        if self.weights is None:
            self.weights = [1]*self.n_classes

        # Prima di ritornare, salva il numero di questa batch
        self.prev_batch_id = batch_id
        self.prev_batch_id_bypass = self.batch_id_bypass

        #print(f"$DG: {self.weights}")
        if self.return_weights:
            if self.concat_output:
                return np.concatenate((X1, X2),axis=-1), Y, self.weights
            else:
                return (X1,X2), Y, self.weights
        else:
            #print("returning data")
            #mask = Dropout(0.1)(np.ones_like(X[...,0:1]), training=True)
            #X = np.array(X)*np.array(mask)
            if self.concat_output:
                return np.concatenate((X1, X2), axis=-1), Y
            else:
                return (X1,X2), Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

        # Reset pivotting indexes
        self.cube_batch_id = 0
        self.prev_batch_id = -1
        self.prev_batch_id_bypass = -1
        self.batch_id_bypass = -1
        #self.buffer_last_index = len(self.X_buffer)

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs_repeated)

    def __fill_buffer(self, list_IDs_temp):

        #print(f"\r{tc.green}Datagenerator '{self.name}'{tc.end}: Filling the buffer with {len(list_IDs_temp)} new samples", end="")

        self.X_path_buffer = []
        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")
        self.buffer_last_index = 0
        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            lowlim = i*self.tiles_per_image
            uprlim = (i+1)*self.tiles_per_image

            # Load sample
            if self.list_type == "path":
                x1 = np.load(self.X1_path_list[ID], mmap_mode="r")
                x2 = np.load(self.X2_path_list[ID], mmap_mode="r")
                #x = resize(x, (self.image_shape), mode='constant')
                #x = x / np.max(x) # Normalization (already done in preprocessing)
            elif self.list_type == "array":
                x1 = self.X1_path_list[ID]
                x2 = self.X2_path_list[ID]

            # Load ground truth
            if self.list_type == "path":
                #print(f"v2_{ID} - {self.X_path_list[ID]}", end="\r")
                y = np.load(self.Y_path_list[ID], mmap_mode="r")
                #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            elif self.list_type == "array":
                y = self.Y_path_list[ID]

            # Apply random augmentation
            if self.apply_augmentation:
                print("Augmentation not available")
                x1,y = dataAugmentationGenerator(x1, y, self.aug_prob, out_dtype=self.dtype)

            # Extract subcubes and fill buffer
            if self.tiles_per_image > 1:
                self.X1_buffer[lowlim:uprlim] = extract_tiles(x1, self.tile_shape, self.stride)
                self.X2_buffer[lowlim:uprlim] = extract_tiles(x2, self.tile_shape, self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_tiles(y, self.tile_shape, self.stride)
            else:
                self.X1_buffer[lowlim:uprlim] = x1
                self.X2_buffer[lowlim:uprlim] = x2
                #self.X_path_buffer.append(self.X1_path_list[ID])
                self.Y_buffer[lowlim:uprlim] = y

            self.buffer_last_index = uprlim

        # Shuffle
        if self.shuffle:
           (self.X1_buffer[0:self.buffer_last_index],
            self.X2_buffer[0:self.buffer_last_index],
            self.Y_buffer[0:self.buffer_last_index]) = (
               shuffle(self.X1_buffer[0:self.buffer_last_index],
                       self.X2_buffer[0:self.buffer_last_index],
                       self.Y_buffer[0:self.buffer_last_index]))

        """if self.apply_balancing:
            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer),
                                                                   verbose=False)
            while len(X_valid) < len(self.X_buffer)//2:
                #print(f"Buffer filled with {len(X_valid)}/{len(self.X_buffer)//2}")
                lm = (len(self.X_buffer)//2) - len(X_valid)
                if lm > len(X_valid):
                    lm = len(X_valid)
                #print(f"Adding {lm} subcubes")
                for i in range(0,lm):
                    x, y = image_augmentation(X_valid[i], Y_valid[i], 0.7)
                    if np.count_nonzero(y)>10:
                        X_valid.append(x)
                        Y_valid.append(y)

            v_id = 0
            e_id = 0
            self.X_buffer *= 0
            self.Y_buffer *= 0
            for i in range(0, len(self.X_buffer)):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1"""

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """
        #for i in subs_indexes:
        #    print(self.X_path_buffer[i] end="\r")
        X1 = self.X1_buffer[subs_indexes]
        X2 = self.X2_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X1,X2, Y

    def get_dataset(self, apply_decomposition, apply_flattening, range=None):
        X1 = []
        X2 = []
        Y = []
        list_IDs = np.copy(self.list_IDs)
        if range is not None:
            list_IDs = list_IDs[range[0]:range[1]]
        for i, ID in enumerate(list_IDs):
            # Load sample
            x1 = np.load(self.X1_path_list[ID], mmap_mode='r')
            x2 = np.load(self.X2_path_list[ID], mmap_mode='r')
            #x = resize(x, (self.image_shape), mode='constant')
            #x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x1 = extract_tiles(x1, (*self.tile_shape,), self.stride)
                x2 = extract_tiles(x2, (*self.tile_shape,), self.stride)
                y = extract_tiles(y, (*self.tile_shape,), self.stride)
            if apply_flattening and apply_decomposition:
                for x1_sub,x2_sub, y_sub in zip(x1,x2,y):
                    X1.append(x1_sub)
                    X2.append(x2_sub)
                    Y.append(y_sub)
            else:
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
        X1 = np.array(X1)
        X2 = np.array(X2)
        Y = np.array(Y)


        return X1,X2,Y


class DataGeneratorSAT_CLA(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    # TODO: Introduce Support to:
    #   X, Y come array invece che come path
    #
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, tile_shape=(128,128),  n_classes=20,
                 n_img_channels=3, n_msk_channels=20, shuffle=True, image_shape=(1920, 960), stride=32, n_img=4, n_aug=5,
                 dtype=np.float16, list_type="path", apply_augmentation=True, apply_balancing=False,
                 allow_partilly_filled_batch=False, force_tensor_conversion=True, return_weights=False, weights=None):

        """ Initialization """
        # Preprocessing params
        self.image_shape = image_shape
        self.tile_shape = tile_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing
        self.dtype = dtype
        self.weights = weights

        # File Loading params
        self.n_img = n_img
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list
        self.list_type = list_type

        # Model params
        self.batch_size = batch_size
        self.n_img_channels = n_img_channels
        self.n_msk_channels = n_msk_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.allow_partilly_filled_batch = allow_partilly_filled_batch
        self.force_tensor_conversion = force_tensor_conversion
        self.return_weights = return_weights
        self.tiles_per_image = compute_tiles_count(image_shape, tile_shape, stride)
        self.tiles_per_buffer = self.tiles_per_image * self.n_img
        self.X_buffer = np.empty((self.tiles_per_buffer, *self.tile_shape, self.n_img_channels), dtype=self.dtype)
        self.Y_buffer = np.empty((self.tiles_per_buffer, self.n_msk_channels), dtype=self.dtype)
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        if self.allow_partilly_filled_batch:
            # Include partially filled batch
            return int(np.ceil((len(self.list_IDs)*self.tiles_per_image*self.n_aug ) / self.batch_size))
        else:
            # Exclude partially filled batch (Default)
            return int(np.floor((len(self.list_IDs) * self.tiles_per_image * self.n_aug) / self.batch_size))



    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = (batch_id) % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_img:(self.cube_batch_id + 1) * self.n_img]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_img > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        if self.force_tensor_conversion:
            with tf.device('/cpu:0'):
                if tf.is_tensor(X) == False:
                    X = tf.convert_to_tensor(X, self.dtype)
                    Y = tf.convert_to_tensor(Y, self.dtype)

        if self.weights is None:
            self.weights = [1]*self.n_classes

        #print(f"$DG: {self.weights}")
        if self.return_weights:
            return X, Y, self.weights
        else:
            #print(X.shape)
            #print(Y.shape)
            return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    def __fill_buffer(self, list_IDs_temp):

        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")

        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            lowlim = i*self.tiles_per_image
            uprlim = (i+1)*self.tiles_per_image

            # Load sample
            if self.list_type == "path":
                x = np.load(self.X_path_list[ID], mmap_mode='r')
                #x = resize(x, (self.image_shape), mode='constant')
                #x = x / np.max(x) # Normalization (already done in preprocessing)
            elif self.list_type == "array":
                x = self.X_path_list[ID]

            # Load ground truth
            if self.list_type == "path":
                y = self.Y_path_list[ID]
                #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)
            elif self.list_type == "array":
                y = self.Y_path_list[ID]

            # Apply random augmentation
            #if self.apply_augmentation:
            #    x,y = dataAugmentationGenerator(x, y, 1-(1/self.n_aug), out_dtype=self.dtype)

            # Extract subcubes and fill buffer
            if self.tiles_per_image > 1:
                self.X_buffer[lowlim:uprlim] = extract_tiles(x, self.tile_shape, self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_tiles(y, self.tile_shape, self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.Y_buffer[lowlim:uprlim] = y

        # Shuffle
        if self.shuffle:
           self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        """if self.apply_balancing:
            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer),
                                                                   verbose=False)
            while len(X_valid) < len(self.X_buffer)//2:
                #print(f"Buffer filled with {len(X_valid)}/{len(self.X_buffer)//2}")
                lm = (len(self.X_buffer)//2) - len(X_valid)
                if lm > len(X_valid):
                    lm = len(X_valid)
                #print(f"Adding {lm} subcubes")
                for i in range(0,lm):
                    x, y = image_augmentation(X_valid[i], Y_valid[i], 0.7)
                    if np.count_nonzero(y)>10:
                        X_valid.append(x)
                        Y_valid.append(y)

            v_id = 0
            e_id = 0
            self.X_buffer *= 0
            self.Y_buffer *= 0
            for i in range(0, len(self.X_buffer)):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1"""

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

    def get_dataset(self, apply_decomposition, apply_flattening):
        X = []
        Y = []
        for i, ID in enumerate(self.list_IDs):
            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            #x = resize(x, (self.image_shape), mode='constant')
            #x = x / np.max(x)  # Normalization

            # Load ground truth
            y = self.Y_path_list[ID]
            #y = resize(y, (self.image_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_tiles(x, (*self.tile_shape,), self.stride)
                y = extract_tiles(y, (*self.tile_shape,), self.stride)
            if apply_flattening and apply_decomposition:
                for x_sub, y_sub in zip(x,y):
                    X.append(x_sub)
                    Y.append(y_sub)
            else:
                X.append(x)
                Y.append(y)
        X = np.array(X)
        Y = np.array(Y)
        return X,Y
