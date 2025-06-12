from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, AveragePooling3D, Conv3DTranspose, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution3D, UpSampling3D, Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Activation, Concatenate, Add, LeakyReLU, Maximum
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU

""""
V4 Backup:
Questa versione è interessante, e raggiunge ottimi livelli di accuratezza
Ogni blocco convolutivo applica due convoluzioni parallele con kernel di diverse dimensioni
"""
smooth = 1.


def decoderBlock(input, skip_features, num_filters, kernel_size):
    up = Conv3DTranspose(num_filters, (3, 3, 3), strides=2, padding="same", kernel_initializer="he_uniform")(input)
    up = BatchNormalization()(up)
    up = Activation(activation="relu")(up)
    merge = Concatenate()([up, skip_features])
    merge = Dropout(0.1)(merge)
    conv_block = convBlock(merge, num_filters, kernel_size)
    return conv_block


def encoderBlock(input, num_filters, kernel_size):
    conv_block = convBlock(input, num_filters, kernel_size)
    pool = MaxPooling3D(pool_size=(2, 2, 2))(conv_block)
    pool = Dropout(0.1)(pool)
    return pool, conv_block


def bridgeBlock(input, num_filters, kernel_size):
    conv_block = convBlock(input, num_filters, kernel_size)
    return conv_block

def convLayer(input, num_filters, k_size):
    conv = Convolution3D(filters=num_filters, kernel_size=k_size, padding='same', kernel_initializer="he_uniform")(input)
    conv = BatchNormalization()(conv)
    conv = Activation(activation="relu")(conv)
    return conv

def convBlock(input, num_filters, kernel_size):
    num_filters = num_filters / 2

    # CONV 1
    conv1_1 = convLayer(input, num_filters, 1)
    #conv1_2 = convLayer(conv1_1, num_filters, 1)

    # CONV 2
    conv2_1 = convLayer(conv1_1, num_filters, 3)
    conv2_2 = convLayer(conv2_1, num_filters, 3)

    # CONV 3
    conv3_1 = convLayer(conv1_1, num_filters, 5)
    conv3_2 = convLayer(conv3_1, num_filters, 5)

    # CONV 4
    #conv4_1 = convLayer(input, num_filters, 7)
    #conv4_2 = convLayer(conv4_1, num_filters, 7)

    merge = Concatenate()([conv2_2, conv3_2])

    #dense = Dense(num_filters * 2, activation=None, kernel_initializer="he_uniform")(merge)
    #dense = BatchNormalization()(dense)
    #dense = Activation(activation="relu")(dense)

    return merge


def build_model(inp_shape, out_channels=1):
    k_size = 3
    filters = 8

    # Input
    data = Input(shape=inp_shape)

    # Downsample Phase
    enc1, conv1 = encoderBlock(data, filters * 1, k_size)
    enc2, conv2 = encoderBlock(enc1, filters * 2, k_size)
    enc3, conv3 = encoderBlock(enc2, filters * 4, k_size)
    enc4, conv4 = encoderBlock(enc3, filters * 4, k_size)

    ## Bridge
    #brdg = bridgeBlock(enc4, filters * 8, k_size)

    # Upsample Phase
    dec1 = decoderBlock(enc4, conv4, filters * 4, k_size)
    dec2 = decoderBlock(dec1, conv3, filters * 4, k_size)
    dec3 = decoderBlock(dec2, conv2, filters * 2, k_size)
    dec4 = decoderBlock(dec3, conv1, filters * 1, k_size)

    """up = Conv2DTranspose(filters, (3, 3), strides=2, padding="same", kernel_initializer="he_normal")(dec3)
    up = BatchNormalization()(up)
    up = Activation(activation="relu")(up)

    merge = Concatenate()([up, dec4])"""
    #conv_preOut = Convolution2D(filters=filters, kernel_size=k_size, padding='same', kernel_initializer="normal")(dec4)
    output = Convolution3D(filters=out_channels, kernel_size=k_size, padding='same',
                           kernel_initializer="normal", activation='sigmoid')(dec4)

    model = Model(inputs=[data], outputs=[output])
    return model

