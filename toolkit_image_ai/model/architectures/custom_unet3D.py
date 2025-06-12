import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Concatenate

def unet_3d(input_shape, num_classes):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv3D(16, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Bottleneck
    conv4 = Conv3D(128, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, 3, activation='relu', padding='same')(conv4)
    drop4 = tf.nn.dropout(conv4, rate=0.5)

    # Decoder

    up7 = Conv3D(64, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(drop4))
    merge7 = Concatenate()([conv3, up7])
    conv7 = Conv3D(64, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(64, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(32, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate()([conv2, up8])
    conv8 = Conv3D(32, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(32, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate()([conv1, up9])
    conv9 = Conv3D(16, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(16, 3, activation='relu', padding='same')(conv9)
    conv9 = Conv3D(num_classes, 1, activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv9)

    return model