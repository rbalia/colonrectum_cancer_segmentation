import re
import time

import tensorflow as tf
import numpy as np
#import cupy as cp
#import cupyx.scipy
from keras.utils import to_categorical
from skimage.transform import resize
from sklearn.utils import shuffle

from toolkit_image_ai.processing.augmentation3D import image_augmentation
from toolkit_image_ai.dataset.decomposition3D_2 import extract_subcubes, compute_subcubes_count
from src_analisys_modules.preprocessing import filter_empty_cubes, split_valid_empty, count_ditribution


class DataGeneratorAOU_fromList(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Get data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(self.X_path_list[ID])#, mmap_mode='r')
            x = x/np.max(x)
            X[i,] = x
            #numpy_data_memmap_1.item(ID)

            try:
                y = np.load(self.Y_path_list[ID])  # , mmap_mode='r')
                #y = y / np.max(y)
            except:
                y = np.zeros_like(x)
            Y[i,] = y

        return X, Y

class DataGeneratorAOUstadiation(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_list, Y_list, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, gt_type="neoplasia"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.gt_type = gt_type
        self.on_epoch_end()
        self.X_list = X_list
        self.Y_list = Y_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 1))#self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #us = np.load(f"dataset_colonrectum/binaryPkg/single_us/ultrasounds_{ID:03d}.npy")#, mmap_mode='r')
            us = np.load(self.X_list[ID])  # , mmap_mode='r')
            us = us/np.max(us)
            X[i,] = resize(us, self.dim, mode='constant')
            #numpy_data_memmap_1.item(ID)

            y_el = self.Y_list[ID]
            #print(y_el)
            #y_el = to_categorical(y_el, num_classes=self.n_classes)

            y[i] = y_el
            #numpy_data_memmap_2.item(ID)

        return X, y

class DataGeneratorAOUstadiation3Dmask(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_list, Ys_list, Yc_list,  batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, n_aug=1, shuffle=True, gt_type="neoplasia", apply_augmentation=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_aug = n_aug
        self.shuffle = shuffle
        self.apply_augmentation = apply_augmentation
        self.gt_type = gt_type
        self.on_epoch_end()
        self.X_list = X_list
        self.Yc_list = Yc_list
        self.Ys_list = Ys_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.list_IDs)*self.n_aug) / self.batch_size))

    def __getitem__(self, batch_id_):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_id_ = batch_id_ % (len(self.indexes) // self.batch_size)
        indexes = self.indexes[batch_id_*self.batch_size:(batch_id_+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs)) #indici per recuperare gli indici dei cubi
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels + 1))
        yc = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #us = np.load(f"dataset_colonrectum/binaryPkg/single_us/ultrasounds_{ID:03d}.npy")#, mmap_mode='r')
            us = np.load(self.X_list[ID], mmap_mode='r')
            gt = np.load(self.Ys_list[ID], mmap_mode='r')
            us = us/np.max(us)
            us = resize(us, self.dim, mode='constant')
            gt = resize(gt, self.dim, order=0, anti_aliasing=False)
            if self.apply_augmentation:
                us, gt = image_augmentation(us, gt, 0.9)
            X[i, ..., 0] = us[...,0]#resize(us[...,0], self.dim, mode='constant')
            X[i, ..., 1] = gt[...,0]#resize(gt[...,0], self.dim, order=0, anti_aliasing=False)
            #numpy_data_memmap_1.item(ID)

            y_ = to_categorical(self.Yc_list[ID], num_classes=5)



            yc[i] = y_

        #yc = to_categorical(yc, num_classes=5)
        return X, yc

class DataGeneratorAOUstadiation3DNOmask(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_list, Ys_list, Yc_list,  batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, n_aug=1, shuffle=True, gt_type="neoplasia", apply_augmentation=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_aug = n_aug
        self.shuffle = shuffle
        self.apply_augmentation = apply_augmentation
        self.gt_type = gt_type
        self.on_epoch_end()
        self.X_list = X_list
        self.Yc_list = Yc_list
        self.Ys_list = Ys_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor((len(self.list_IDs)*self.n_aug) / self.batch_size))

    def __getitem__(self, batch_id_):
        'Generate one batch of data'
        # Generate indexes of the batch
        batch_id_ = batch_id_ % (len(self.indexes) // self.batch_size)
        indexes = self.indexes[batch_id_*self.batch_size:(batch_id_+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs)) #indici per recuperare gli indici dei cubi
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        yc = np.empty((self.batch_size, self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #us = np.load(f"dataset_colonrectum/binaryPkg/single_us/ultrasounds_{ID:03d}.npy")#, mmap_mode='r')
            us = np.load(self.X_list[ID], mmap_mode='r')
            #gt = np.load(self.Ys_list[ID], mmap_mode='r')
            us = us/np.max(us)
            us = resize(us, self.dim, mode='constant')
            #gt = resize(gt, self.dim, order=0, anti_aliasing=False)
            if self.apply_augmentation:
                us, _ = image_augmentation(us, us, 0.9)
            X[i, ..., 0] = us[...,0]#resize(us[...,0], self.dim, mode='constant')
            #X[i, ..., 1] = gt[...,0]#resize(gt[...,0], self.dim, order=0, anti_aliasing=False)
            #numpy_data_memmap_1.item(ID)

            y_ = to_categorical(self.Yc_list[ID], num_classes=5)



            yc[i] = y_

        #yc = to_categorical(yc, num_classes=5)
        return X, yc

class DataGeneratorAOUstadiation3DmaskTslice(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_list, Ys_list, Yc_list,  batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True, gt_type="neoplasia"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.gt_type = gt_type
        self.on_epoch_end()
        self.X_list = X_list
        self.Yc_list = Yc_list
        self.Ys_list = Ys_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels + 1))
        yc = np.empty((self.batch_size, 96))#self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #us = np.load(f"dataset_colonrectum/binaryPkg/single_us/ultrasounds_{ID:03d}.npy")#, mmap_mode='r')
            us = np.load(self.X_list[ID])  # , mmap_mode='r')
            gt = np.load(self.Ys_list[ID])  # , mmap_mode='r')
            us = us/np.max(us)
            X[i, ..., 0] = us[...,0]
            X[i, ..., 1] = gt[...,0]
            #numpy_data_memmap_1.item(ID)

            y_str = self.Yc_list[ID]
            # Convert csv string to numpy array of tuples (slice_id, t_stage)
            list_of_strings = re.findall(r'\(([^\)]*)\)', y_str)
            list_of_tuples = [tuple(map(int, x.split(','))) for x in list_of_strings]
            array = np.array(list_of_tuples)
            # print(array)

            arr = np.array([])
            prev_index = array[0][0]
            prev_tstage = array[0][1]
            for id, pair in enumerate(array[1:]):
                curr_index = pair[0]
                curr_tstage = pair[1]
                slice_ids = range(prev_index, curr_index)
                range_len = len(slice_ids)
                # t_stages = np.repeat(to_categorical(prev_tstage, num_classes=self.n_classes), range_len)
                t_stages = np.repeat(prev_tstage, range_len)
                # print(t_stages)
                arr = np.concatenate((arr, t_stages))

                prev_index = curr_index
                prev_tstage = curr_tstage

            # convert to 96 element array
            arr = np.array(arr)

            # print("w")
            # print(array.shape)
            arr = resize(arr, (96,), order=0, anti_aliasing=False)
            #print(y_el)
            #y_el = to_categorical(y_el, num_classes=self.n_classes)

            yc[i] = arr
            #numpy_data_memmap_2.item(ID)

        return X, yc

class DataGeneratorAOUstadiation2D(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, X_list, Y_list, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, gt_type="neoplasia"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.gt_type = gt_type
        self.on_epoch_end()
        self.X_list = X_list
        self.Y_list = Y_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 96, ))#self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            #us = np.load(f"dataset_colonrectum/binaryPkg/single_us/ultrasounds_{ID:03d}.npy")#, mmap_mode='r')
            us = np.load(self.X_list[ID])  # , mmap_mode='r')
            us = us/np.max(us)
            X[i,] = us
            #numpy_data_memmap_1.item(ID)

            y_str = self.Y_list[ID]

            # Convert csv string to numpy array of tuples (slice_id, t_stage)
            list_of_strings = re.findall(r'\(([^\)]*)\)', y_str)
            list_of_tuples = [tuple(map(int, x.split(','))) for x in list_of_strings]
            array = np.array(list_of_tuples)
            #print(array)

            arr = np.array([])
            prev_index = array[0][0]
            prev_tstage = array[0][1]
            for id, pair in enumerate(array[1:]):
                curr_index = pair[0]
                curr_tstage = pair[1]
                slice_ids = range(prev_index, curr_index)
                range_len = len(slice_ids)
                #t_stages = np.repeat(to_categorical(prev_tstage, num_classes=self.n_classes), range_len)
                t_stages = np.repeat(prev_tstage, range_len)
                #print(t_stages)
                arr = np.concatenate((arr, t_stages))

                prev_index = curr_index
                prev_tstage = curr_tstage

            # convert to 96 element array
            arr = np.array(arr)

            #print("w")
            #print(array.shape)
            arr = resize(arr, (96,), order=0, anti_aliasing=False)
            #print(arr)
            #y_el = to_categorical(y_el, num_classes=self.n_classes)

            y[i] = arr
            #numpy_data_memmap_2.item(ID)

        return X, y

class DataGeneratorAOU_fixedSize(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=6,
                 apply_augmentation=True):

        """ Initialization """
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(np.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

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
            lowlim = i*self.subs_per_cube
            uprlim = (i+1)*self.subs_per_cube

            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x) # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            self.X_buffer[lowlim:uprlim] = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
            self.Y_buffer[lowlim:uprlim] = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)

        # Shuffle
        if self.shuffle:
            self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

class DataGeneratorAOU_fixedSize_balanced(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=6,
                 apply_augmentation=True):

        """ Initialization """
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.buff_lim = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(np.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) \
                or ((batch_id_ + 1) * self.batch_size > self.buff_lim) \
                or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            #print(f"n cubes: {self.n_cubes}")
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

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

        self.buff_lim = 0

        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            #lowlim = i*self.subs_per_cube
            #uprlim = (i+1)*self.subs_per_cube

            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x) # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)


            #print(f"id: {ID} - file: {self.X_path_list[ID]}")

            #plotter3D.plotVolumetricSlices(x, [y], axis_name=["Ax", "Cor", "Sag"], mask_mean_projection=True)

            # Extract subcubes and fill buffer
            X_subcubes = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
            Y_subcubes = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
            #print(X_subcubes)
            #print(Y_subcubes)
            X_subcubes, Y_subcubes = filter_empty_cubes(X_subcubes, Y_subcubes, verbose=False, shuffle_flag=True)
            filt_count = len(X_subcubes)

            #print(len(X_subcubes))
            #print(len(X_subcubes_filt))
            if filt_count > 0:
                self.X_buffer[self.buff_lim:self.buff_lim+filt_count] = X_subcubes
                self.Y_buffer[self.buff_lim:self.buff_lim+filt_count] = Y_subcubes

                self.buff_lim += filt_count

        while self.buff_lim < len(self.X_buffer):
            print(f"Buffer filled with {self.buff_lim}/{len(self.X_buffer)}")
            lm = len(self.X_buffer) - self.buff_lim
            if lm > self.buff_lim:
                lm = self.buff_lim
            print(f"Adding {lm} subcubes")
            for i in range(0,lm):
                x, y = image_augmentation(self.X_buffer[i], self.Y_buffer[i], 0.7)
                self.X_buffer[self.buff_lim + i] = x
                self.Y_buffer[self.buff_lim + i] = y
            self.buff_lim += lm

        # Shuffle
        if self.shuffle:
            self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        print("FILLING FINISHED")


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

class DataGeneratorAOU_fixedSize_balanced_2(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=6,
                 apply_augmentation=True, apply_balancing=False):

        """ Initialization """
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(np.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

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
            lowlim = i*self.subs_per_cube
            uprlim = (i+1)*self.subs_per_cube

            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x) # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            self.X_buffer[lowlim:uprlim] = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
            self.Y_buffer[lowlim:uprlim] = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)

        empty_cubes_pre, valid_cubes_pre = count_ditribution(self.X_buffer, self.Y_buffer, verbose=True)
        #X_valid, Y_valid, X_empty, Y_empty = split_valid_empty(self.X_buffer, self.Y_buffer, verbose=False)
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
                e_id += 1

        # Shuffle
        #if self.shuffle:
        #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

class DataGeneratorAOU_fixedSize_balanced_3(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=6,
                 apply_augmentation=True, apply_balancing=False):

        """ Initialization """
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_balancing = apply_balancing

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(np.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

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
            lowlim = i*self.subs_per_cube
            uprlim = (i+1)*self.subs_per_cube

            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x) # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            if self.subs_per_cube > 1:
                self.X_buffer[lowlim:uprlim] = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.Y_buffer[lowlim:uprlim] = y

        # Shuffle
        if self.shuffle:
           self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        if self.apply_balancing:
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
                    e_id += 1

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
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
                y = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
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

class DataGeneratorAOU_fixedSize_balanced_4Cupy(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    Inoltre gestisce tutto il dataset in fase di addestramento e validazione, implementando funzioni per il caricamento
    dei dati, il preprocessing, l'augmentation e il bilanciamento del numero di campioni.
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=1, shuffle=False, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=1,
                 apply_augmentation=False, apply_augmentation_balancing=False, apply_filtration_balancing=False,
                 preload_samples=False, verbose_debug=False):

        """ Initialization """
        self.verbose_debug = verbose_debug
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_augmentation_balancing = apply_augmentation_balancing
        self.apply_filtration_balancing = apply_filtration_balancing
        self.preload_samples = preload_samples

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = cp.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = cp.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.buffer_samples = self.subs_per_buffer #Edit if the buffer is not filled (when using balancing by filtering)

        if self.preload_samples:
            self._preload_samples()

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(cp.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        if self.apply_filtration_balancing:
            batch_id_ = batch_id % (self.buffer_samples // self.batch_size)
        else:
            batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) \
                or ((batch_id_ + 1) * self.batch_size > self.buffer_samples) \
                or batch_id_ == 0:

            if self.verbose_debug: print(f"Refill Buffer - batch_id_:{batch_id_} "
                                         f"- batch_id:{batch_id} "
                                         f"- buffer_limit:{self.buffer_samples}")

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = cp.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    def __fill_buffer(self, list_IDs_temp):

        if self.verbose_debug: print(f"Filling Buffer - current cubes:{list_IDs_temp} - pool:{self.list_IDs}")
        start = time.time()
        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")

        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            if self.verbose_debug: print(f"Filling Buffer - current cube:{ID}")
            lowlim = i*self.subs_per_cube
            uprlim = (i+1)*self.subs_per_cube

            # Load sample
            if self.preload_samples:
                x = self.X_path_list[ID]
            else:
                x = cp.load(self.X_path_list[ID], mmap_mode='r')
                x = resize(cp.asnumpy(x), (self.full_cube_shape), mode='constant')
                x = x / np.max(x) # Normalization

            # Load ground truth
            if self.preload_samples:
                y = self.Y_path_list[ID]
            else:
                y = cp.load(self.Y_path_list[ID], mmap_mode='r')
                y = resize(cp.asnumpy(y), (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            if self.subs_per_cube > 1:
                self.X_buffer[lowlim:uprlim] = cp.asarray(extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride))
                self.Y_buffer[lowlim:uprlim] = cp.asarray(extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride))
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.Y_buffer[lowlim:uprlim] = y

        # Shuffle
        if self.shuffle:
           self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        if self.apply_augmentation_balancing or self.apply_filtration_balancing:

            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = \
                split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer), verbose=False)

            if self.apply_augmentation_balancing:
                if self.verbose_debug: print(f"Apply balancing by Augmentation")
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

            if self.apply_filtration_balancing:
                if self.verbose_debug: print(f"Apply balancing by Filtration")
                if len(X_valid) < len(X_empty):
                    lowlim = len(X_valid)
                else:
                    lowlim = len(X_empty)
                self.buffer_samples = lowlim * 2

            # Initializes indexes of valid and empty arrays
            v_id = 0
            e_id = 0

            # Reset buffer
            self.X_buffer *= 0
            self.Y_buffer *= 0

            # Fill the buffer (first #self.buffer_samples positions. If balancing is applied by Augmentation, the value
            # is equal to the maximum size of the buffer, and it is initialize during the class creation, otherwise if
            # the Filtration Balancing is applied, the value is rewrite just above, according the number of valid cubes)
            for i in range(0, self.buffer_samples):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1


        if self.verbose_debug: print(f"Buffer Refilled - buffer_limit:{self.buffer_samples} - Time:{time.time()-start}")

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = cp.asnumpy(self.X_buffer[subs_indexes])
        Y = cp.asnumpy(self.Y_buffer[subs_indexes])

        #X = tf.convert_to_tensor(self.X_buffer[subs_indexes])
        #Y = tf.convert_to_tensor(self.Y_buffer[subs_indexes])

        #Xa = self.X_buffer[subs_indexes]
        #Ya = self.Y_buffer[subs_indexes]
        #Xb = X.toDlpack()
        #Yb = Y.toDlpack()
        #X = tf.experimental.dlpack.from_dlpack(Xa)
        #Y = tf.experimental.dlpack.from_dlpack(Ya)

        return X, Y

    def _preload_samples(self):

        if self.verbose_debug: print(f"Preloading Samples")

        new_list_IDs = []
        X_samples = []
        Y_samples = []

        for i, ID in enumerate(self.list_IDs):
            # Load sample
            x = cp.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(cp.asnumpy(x), (self.full_cube_shape), mode='constant')
            x = x / np.max(x)  # Normalization

            # Load ground truth
            y = cp.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(cp.asnumpy(y), (self.full_cube_shape), order=0, anti_aliasing=False)

            X_samples.append(x)
            Y_samples.append(y)
            new_list_IDs.append(i)

        self.list_IDs = new_list_IDs
        self.X_path_list = X_samples
        self.Y_path_list = Y_samples

    def get_dataset(self, apply_decomposition, apply_flattening):
        X = []
        Y = []
        for i, ID in enumerate(self.list_IDs):
            # Load sample
            if self.preload_samples:
                x = self.X_path_list[ID]
            else:
                x = cp.load(self.X_path_list[ID], mmap_mode='r')
                x = resize(cp.asnumpy(x), (self.full_cube_shape), mode='constant')
                x = x / np.max(x)  # Normalization

            # Load ground truth
            if self.preload_samples:
                y = self.Y_path_list[ID]
            else:
                y = cp.load(self.Y_path_list[ID], mmap_mode='r')
                y = resize(cp.asnumpy(y), (self.full_cube_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
                y = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
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

class DataGeneratorAOU_fixedSize_balanced_4(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria
    Inoltre gestisce tutto il dataset in fase di addestramento e validazione, implementando funzioni per il caricamento
    dei dati, il preprocessing, l'augmentation e il bilanciamento del numero di campioni.
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=1, shuffle=False, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=1,
                 apply_augmentation=False, apply_augmentation_balancing=False, apply_filtration_balancing=False,
                 preload_samples=False, verbose_debug=False):

        """ Initialization """
        self.verbose_debug = verbose_debug
        # Preprocessing params
        self.full_cube_shape = full_cube_shape
        self.subs_cube_shape = subs_cube_shape
        self.stride = stride
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.apply_augmentation_balancing = apply_augmentation_balancing
        self.apply_filtration_balancing = apply_filtration_balancing
        self.preload_samples = preload_samples

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_cube = compute_subcubes_count(full_cube_shape, subs_cube_shape, stride)
        self.subs_per_buffer = self.subs_per_cube * self.n_cubes
        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.buffer_samples = self.subs_per_buffer #Edit if the buffer is not filled (when using balancing by filtering)

        if self.preload_samples:
            self.__preload_samples()

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations

        # Exclude partially filled batch
        return int(np.floor((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size))

        # Include partially filled batch
        #return ceil((len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size)

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        if self.apply_filtration_balancing:
            batch_id_ = batch_id % (self.buffer_samples // self.batch_size)
        else:
            batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) \
                or ((batch_id_ + 1) * self.batch_size > self.buffer_samples) \
                or batch_id_ == 0:

            if self.verbose_debug: print(f"Refill Buffer - batch_id_:{batch_id_} "
                                         f"- batch_id:{batch_id} "
                                         f"- buffer_limit:{self.buffer_samples}")

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to read samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

        return X, Y

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.update_cubes_indexes()

    def update_cubes_indexes(self):
        """ Update the indexes of cubes to load """
        if self.shuffle == True:
            np.random.shuffle(self.list_IDs)

    def __fill_buffer(self, list_IDs_temp):

        if self.verbose_debug: print(f"Filling Buffer - current cubes:{list_IDs_temp} - pool:{self.list_IDs}")
        start = time.time()
        #print(f"Filling Buffer. "
        #      f"\n\tUsing cubes with IDs: {list_IDs_temp} "
        #      f"\n\tPool of cubes with IDs: {self.list_IDs}")

        # Load selected cubes
        for i, ID in enumerate(list_IDs_temp):
            if self.verbose_debug: print(f"Filling Buffer - current cube:{ID}")
            lowlim = i*self.subs_per_cube
            uprlim = (i+1)*self.subs_per_cube

            # Load sample
            if self.preload_samples:
                x = self.X_path_list[ID]
            else:
                x = np.load(self.X_path_list[ID], mmap_mode='r')
                x = resize(x, (self.full_cube_shape), mode='constant')
                x = x / np.max(x) # Normalization

            # Load ground truth
            if self.preload_samples:
                y = self.Y_path_list[ID]
            else:
                y = np.load(self.Y_path_list[ID], mmap_mode='r')
                y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            if self.subs_per_cube > 1:
                self.X_buffer[lowlim:uprlim] = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
                self.Y_buffer[lowlim:uprlim] = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
            else:
                self.X_buffer[lowlim:uprlim] = x
                self.Y_buffer[lowlim:uprlim] = y

        # Shuffle
        if self.shuffle:
           self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)

        if self.apply_augmentation_balancing or self.apply_filtration_balancing:

            #count_ditribution(self.X_buffer, self.Y_buffer, verbose=False)
            X_valid, Y_valid, X_empty, Y_empty = \
                split_valid_empty(np.copy(self.X_buffer), np.copy(self.Y_buffer), verbose=False)

            if self.apply_augmentation_balancing:
                if self.verbose_debug: print(f"Apply balancing by Augmentation")
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

            if self.apply_filtration_balancing:
                if self.verbose_debug: print(f"Apply balancing by Filtration")
                if len(X_valid) < len(X_empty):
                    lowlim = len(X_valid)
                else:
                    lowlim = len(X_empty)
                self.buffer_samples = lowlim * 2

            # Initializes indexes of valid and empty arrays
            v_id = 0
            e_id = 0

            # Reset buffer
            self.X_buffer *= 0
            self.Y_buffer *= 0

            # Fill the buffer (first #self.buffer_samples positions. If balancing is applied by Augmentation, the value
            # is equal to the maximum size of the buffer, and it is initialize during the class creation, otherwise if
            # the Filtration Balancing is applied, the value is rewrite just above, according the number of valid cubes)
            for i in range(0, self.buffer_samples):#imV, imE, msV, msE in zip(X_valid, X_empty, Y_valid, Y_empty):
                if i%2:
                    self.X_buffer[i] = X_valid[v_id]
                    self.Y_buffer[i] = Y_valid[v_id]
                    v_id += 1
                else:
                    self.X_buffer[i] = X_empty[e_id]
                    self.Y_buffer[i] = Y_empty[e_id]
                    e_id += 1


        if self.verbose_debug: print(f"Buffer Refilled - buffer_limit:{self.buffer_samples} - Time:{time.time()-start}")

            ## Shuffle
            #if self.shuffle:
            #    self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y

    def __preload_samples(self):

        if self.verbose_debug: print(f"Preloading Samples")

        new_list_IDs = []
        X_samples = []
        Y_samples = []

        for i, ID in enumerate(self.list_IDs):
            # Load sample
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            x = resize(x, (self.full_cube_shape), mode='constant')
            x = x / np.max(x)  # Normalization

            # Load ground truth
            y = np.load(self.Y_path_list[ID], mmap_mode='r')
            y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            X_samples.append(x)
            Y_samples.append(y)
            new_list_IDs.append(i)

        self.list_IDs = new_list_IDs
        self.X_path_list = X_samples
        self.Y_path_list = Y_samples

    def get_dataset(self, apply_decomposition, apply_flattening):
        X = []
        Y = []
        #sorted_IDs = list(self.list_IDs).sort()
        print(self.list_IDs)
        for i, ID in enumerate(sorted(self.list_IDs)):
            # Load sample
            if self.preload_samples:
                x = self.X_path_list[ID]
            else:
                x = np.load(self.X_path_list[ID], mmap_mode='r')
                x = resize(x, (self.full_cube_shape), mode='constant')
                x = x / np.max(x)  # Normalization

            # Load ground truth
            if self.preload_samples:
                y = self.Y_path_list[ID]
            else:
                y = np.load(self.Y_path_list[ID], mmap_mode='r')
                y = resize(y, (self.full_cube_shape), order=0, anti_aliasing=False)

            if apply_decomposition:
                x = extract_subcubes(x, (*self.subs_cube_shape, 1), self.stride)
                y = extract_subcubes(y, (*self.subs_cube_shape, 1), self.stride)
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


class DataGeneratorAOU_multiConf(tf.keras.utils.Sequence):
    """
    Generatore di dati per modelli Keras.
    Carica un numero controllato di cubi per volta per evitare di saturare la memoria.
    """
    def __init__(self, list_IDs, X_path_list, Y_path_list, batch_size=32, subs_cube_shape=(64,64,64), n_channels=1,
                 n_classes=10, shuffle=True, full_cube_shape=(160,160,160,1), stride=32, n_cubes=4, n_aug=10,
                 apply_augmentation=True, extraction_setting=[{'cube_shape':(128,128,128,1), 'stride':32}]):

        """ Initialization """
        # Preprocessing params
        self.subs_cube_shape = subs_cube_shape
        self.shuffle = shuffle
        self.n_aug = n_aug
        self.apply_augmentation = apply_augmentation
        self.config = extraction_setting

        # File Loading params
        self.n_cubes = n_cubes
        self.list_IDs = list_IDs
        self.X_path_list = X_path_list
        self.Y_path_list = Y_path_list

        # Model params
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Data Buffers params
        self.cube_batch_id = 0
        self.subs_per_buffer = 0
        #print(self.subs_per_buffer)
        # Compute Subs-per-cube and Subs-per-buffer considering multiple configurations
        n_conf = len(self.config)
        for i in range(0,self.n_cubes):#. enumerate(self.config):
            j = i%n_conf
            self.config[j]["subs_per_cube"] = \
                compute_subcubes_count(self.config[j]["cube_shape"],self.subs_cube_shape,self.config[j]["stride"])
            self.subs_per_buffer += self.config[j]["subs_per_cube"]
            #print(self.subs_per_buffer)

        self.X_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.Y_buffer = np.empty((self.subs_per_buffer, *self.subs_cube_shape, self.n_channels))
        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """

        batch_count = 0
        for i in range(0,len(self.list_IDs)):#. enumerate(self.config):
            j = i%len(self.config)
            batch_count += self.config[j]["subs_per_cube"]
        batch_count = (batch_count * self.n_aug) / self.batch_size

        # Batch per epoch is computed as N_training_samples * N_subcubes * N_augmentations
        #batch_count = (len(self.list_IDs)*self.subs_per_cube*self.n_aug ) / self.batch_size

        # Exclude partially filled batch
        batch_count = int(np.floor(batch_count))

        # Include partially filled batch
        #batch_count = ceil(batch_count)

        return batch_count

    def __getitem__(self, batch_id):
        """ Generate one batch of data """

        # Edit the batch ID in order to cyclicaly read from buffer
        batch_id_ = batch_id % (len(self.X_buffer) // self.batch_size)

        #print(f"batchID {batch_id} - batchID_ {batch_id_}")

        # Load new samples if the buffer has been consumed
        if ((batch_id_ + 1) * self.batch_size > len(self.X_buffer)) or batch_id_ == 0:

            # Get N cube paths indexes from list of usable samples
            list_IDs_temp = self.list_IDs[self.cube_batch_id * self.n_cubes:(self.cube_batch_id + 1) * self.n_cubes]

            # Fill the buffer
            self.__fill_buffer(list_IDs_temp)
            self.cube_batch_id += 1

            # TODO: Fill with random samples if the batch of cubes is not enough large

            # If the cubes has been consumed
            # or there aren't enough samples to fill the buffer, read again and update indexes (shuffle)
            if (self.cube_batch_id + 1) * self.n_cubes > len(self.list_IDs):
                self.cube_batch_id = 0
                self.update_cubes_indexes()

            # Generate indexes to select samples
            self.subs_cube_indexes = np.arange(len(self.X_buffer))

        # Load the indexes of the elements of the current model batch
        subs_indexes = self.subs_cube_indexes[batch_id_ * self.batch_size:(batch_id_ + 1) * self.batch_size]

        # Load a batch of data from buffer
        X, Y = self.__data_generation(subs_indexes)

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
        lim = 0
        for i, ID in enumerate(list_IDs_temp):
            j = i%len(self.config)
            curr_subs_per_cube = self.config[j]["subs_per_cube"]
            curr_cube_shape = self.config[j]["cube_shape"]
            curr_stride = self.config[j]["stride"]

            # Load samples
            x = np.load(self.X_path_list[ID], mmap_mode='r')
            y = np.load(self.Y_path_list[ID], mmap_mode='r')

            # Resize
            x = resize(x, curr_cube_shape, mode='constant')
            y = resize(y, curr_cube_shape, order=0, anti_aliasing=False)

            # Normalization
            x = x / np.max(x)

            # Apply random augmentation
            if self.apply_augmentation:
                x,y = image_augmentation(x, y, 0.9)

            # Extract subcubes and fill buffer
            self.X_buffer[lim:lim+curr_subs_per_cube] = extract_subcubes(x, (*self.subs_cube_shape, 1), curr_stride)
            self.Y_buffer[lim:lim+curr_subs_per_cube] = extract_subcubes(y, (*self.subs_cube_shape, 1), curr_stride)

            # Scroll index
            lim += curr_subs_per_cube

        # Shuffle
        if self.shuffle:
            self.X_buffer, self.Y_buffer = shuffle(self.X_buffer, self.Y_buffer)


    def __data_generation(self,subs_indexes):
        """ Load a batch of data from buffer """

        X = self.X_buffer[subs_indexes]
        Y = self.Y_buffer[subs_indexes]

        return X, Y