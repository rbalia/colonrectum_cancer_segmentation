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
from tensorflow.keras.layers import Lambda
from tensorflow.python.layers.base import Layer

""""
V4 Backup:
Questa versione è interessante, e raggiunge ottimi livelli di accuratezza
Ogni blocco convolutivo applica due convoluzioni parallele con kernel di diverse dimensioni
"""
smooth = 1.

def decomposedConvolution(input, num_filters, k_size):
    conv_block = input
    num_filters = num_filters
    input_shape = (conv_block.shape[1], conv_block.shape[2], conv_block.shape[3],)
    subcube_size = (conv_block.shape[1] // 2, conv_block.shape[2] // 2, conv_block.shape[3] // 2)
    #stride = (subcube_size[0] // 2, subcube_size[1] // 2, subcube_size[2] // 2)
    stride = subcube_size#(subcube_size[0] // 2, subcube_size[1] // 2, subcube_size[2] // 2)

    #print(f"input shape: {input_shape}")
    #print(f"strid shape: {stride}")
    for x in range(0, input_shape[0] - subcube_size[0] + 1, stride[0]):
        for y in range(0, input_shape[1] - subcube_size[1] + 1, stride[1]):
            for z in range(0, input_shape[2] - subcube_size[2] + 1, stride[2]):
                subcube_cell = conv_block[:, x:x + subcube_size[0], y:y + subcube_size[1], z:z + subcube_size[2], :]
                subcube_cell = Conv3D(num_filters, k_size, padding='same')(subcube_cell)
                #print(f"CELL: {subcube_cell.shape} - x.y.z={x}.{y}.{z}")

                if z==0:
                    subcube_row = subcube_cell
                else:
                    subcube_row = Concatenate(axis=2)([subcube_row,subcube_cell])
                #print(f"ROW: {subcube_row.shape} - x.y.z={x}.{y}.{z}")

            if y == 0:
                subcube_plane = subcube_row
            else:
                subcube_plane = Concatenate(axis=3)([subcube_plane, subcube_row])
            #print(f"PLANE: {subcube_plane.shape} - x.y.z={x}.{y}.{z}")

        if x == 0:
            subcube_cube = subcube_plane
        else:
            subcube_cube = Concatenate(axis=1)([subcube_cube, subcube_plane])
        #print(f"CUBE: {subcube_cube.shape} - x.y.z={x}.{y}.{z}")
    #print("\n")

    #subcube_cube = Conv3D(1, 1, padding='same')(subcube_cube)
    subcube_cube = BatchNormalization()(subcube_cube)
    subcube_cube = Activation(activation="relu")(subcube_cube)
    return subcube_cube


def decoderBlock(input, skip_features, num_filters, kernel_size):
    up = Conv3DTranspose(num_filters, (3, 3, 3), strides=2, padding="same", kernel_initializer="he_uniform")(input)
    up = BatchNormalization()(up)
    up = Activation(activation="relu")(up)
    merge = Concatenate()([up, skip_features])
    merge = Dropout(0.1)(merge)
    conv_block = convBlock(merge, num_filters, kernel_size)
    conv_block_decomp = decomposedConvolution(merge, num_filters, kernel_size)
    merge2 = Concatenate()([conv_block, conv_block_decomp])
    return merge2


def encoderBlock(input, num_filters, kernel_size):
    conv_block = convBlock(input, num_filters, kernel_size)
    conv_block_decomp = decomposedConvolution(input, num_filters, kernel_size)
    merge = Add()([conv_block, conv_block_decomp])
    pool = MaxPooling3D(pool_size=(2, 2, 2))(merge)
    pool = Dropout(0.1)(pool)
    return pool, merge

def bridgeBlock(input, num_filters, kernel_size):
    conv_block = convBlock(input, num_filters, kernel_size)
    return conv_block

def convLayer(input, num_filters, k_size, stride=1, dilation=1):
    conv = Convolution3D(filters=num_filters, kernel_size=k_size, strides=stride, dilation_rate=dilation,
                         padding='same', kernel_initializer="he_uniform")(input)
    conv = BatchNormalization()(conv)
    conv = Activation(activation="relu")(conv)
    return conv

def deconvLayer(input, num_filters, k_size, stride=1, dilation=1):
    conv = Conv3DTranspose(num_filters, k_size, strides=stride, padding="same", kernel_initializer="he_uniform")(input)
    conv = BatchNormalization()(conv)
    conv = Activation(activation="relu")(conv)
    return conv

def convBlock(input, num_filters, kernel_size):
    num_filters = num_filters

    # CONV 1
    conv1_1 = convLayer(input, num_filters, 1)

    # CONV 2
    conv2_1 = convLayer(conv1_1, num_filters, 3)
    conv2_2 = convLayer(conv2_1, num_filters, 3)

    # CONV 3
    conv3_1 = convLayer(conv1_1, num_filters, 5)
    conv3_2 = convLayer(conv3_1, num_filters, 5)

    merge = Add()([conv2_2, conv3_2])

    merge = convLayer(merge, num_filters, 3)

    return merge


def build_model(inp_shape, out_channels=1):
    k_size = 3
    filters = 4

    # Input
    data = Input(shape=inp_shape)
    conv1 = convLayer(data, filters, k_size, stride=1, dilation=1)

    # Encoder/Downsampling
    conv2 = convLayer(conv1, filters * 2, k_size, stride=2, dilation=1)
    conv3 = convLayer(conv2, filters * 4, k_size, stride=2, dilation=1)

    # Bridge
    bridge1 = convLayer(conv3,   filters * 4, k_size, stride=1, dilation=2)
    bridge2 = convLayer(bridge1, filters * 4, k_size, stride=1, dilation=4)
    bridge3 = convLayer(bridge2, filters * 4, k_size, stride=1, dilation=8)
    bridge4 = convLayer(bridge3, filters * 4, k_size, stride=1, dilation=1)

    # Decoding/Upsampling
    deconv1 = deconvLayer(bridge4, filters * 4, k_size, stride=2, dilation=1)
    deconv2 = deconvLayer(deconv1, filters * 2, k_size, stride=2, dilation=1)

    # Skip Connection
    conv1_2 = convLayer(conv1, filters*2, k_size, stride=1, dilation=1)
    skipcon = Add()([conv1_2, deconv2])
    """# Downsample Phase
    enc1, conv1 = encoderBlock(data, filters * 1, k_size)
    enc2, conv2 = encoderBlock(enc1, filters * 4, k_size)
    enc3, conv3 = encoderBlock(enc2, filters * 8, k_size)
    enc4, conv4 = encoderBlock(enc3, filters * 16, k_size)

    # Upsample Phase
    dec1 = decoderBlock(enc4, conv4, filters * 16, k_size)
    dec2 = decoderBlock(dec1, conv3, filters * 8, k_size)
    dec3 = decoderBlock(dec2, conv2, filters * 4, k_size)
    dec4 = decoderBlock(dec3, conv1, filters * 1, k_size)"""

    output = Convolution3D(filters=out_channels, kernel_size=k_size, padding='same',
                           kernel_initializer="normal", activation='sigmoid')(skipcon)

    model = Model(inputs=[data], outputs=[output])
    return model
