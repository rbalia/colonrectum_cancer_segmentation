from glob import glob
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import MaxPool2D, Dropout, InputLayer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50, VGG19, DenseNet121
#from tensorflow.python.keras.applications import ResNet50, VGG19, DenseNet121
import tensorflow as tf

from src import configs as conf


smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def convBlock(input, num_filters, kernel_size):
    conv1 = Conv2D(num_filters, kernel_size, padding="same")(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    conv2 = Conv2D(num_filters, kernel_size, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    return conv2

def decoderBlock(input, skip_features, num_filters, kernel_size=3):
    up = Conv2DTranspose(num_filters, (2, 2), strides=2,padding="same")(input)
    merge = Concatenate()([up, skip_features])
    conv_block = convBlock(merge, num_filters, kernel_size)
    return conv_block

def build_unet_resnet(input_shape, weights="imagenet"):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights=weights, input_tensor=inputs)

    # Freeze all Layers
    """for i in range(0, len(resnet50.layers)):
        resnet50.layers[i].trainable = False"""

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output  ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output  ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

    """ Decoder """
    d1 = decoderBlock(b1, s4, 512)  ## (64 x 64)
    d2 = decoderBlock(d1, s3, 256)  ## (128 x 128)
    d3 = decoderBlock(d2, s2, 128)  ## (256 x 256)
    d4 = decoderBlock(d3, s1, 64)   ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="ResNet50_U-Net")
    return model


def build_unet_densenet(input_shape, out_channels=1, weights="imagenet"):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained DenseNet121 Model """
    densenet = DenseNet121(include_top=False, weights=weights, input_tensor=inputs)

    # Freeze all Layers
    """for i in range(0, len(densenet.layers)):
        densenet.layers[i].trainable = False"""

    """ Encoder """
    enc1 = densenet.get_layer("input_1").output  # 512
    enc2 = densenet.get_layer("conv1/relu").output  # 256
    enc3 = densenet.get_layer("pool2_relu").output  # 128
    enc4 = densenet.get_layer("pool3_relu").output  # 64

    """ Bridge """
    b1 = densenet.get_layer("pool4_relu").output  ## 32

    """ Decoder """
    dec1 = decoderBlock(b1, enc4, 512)  ## 64
    dec2 = decoderBlock(dec1, enc3, 256)  ## 128
    dec3 = decoderBlock(dec2, enc2, 128)  ## 256
    dec4 = decoderBlock(dec3, enc1, 64)  ## 512

    """ Outputs """
    outputs = Conv2D(out_channels, 1, padding="same", activation="sigmoid")(dec4)

    model = Model(inputs, outputs)
    return model


def build_unet_vgg19(input_shape):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained VGG19 Model """
    vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

    # Freeze all Layers
    """for i in range(0, len(vgg19.layers)):
        vgg19.layers[i].trainable = False"""

    """ Encoder """
    enc1 = vgg19.get_layer("block1_conv2").output  ## (512 x 512)
    enc2 = vgg19.get_layer("block2_conv2").output  ## (256 x 256)
    enc3 = vgg19.get_layer("block3_conv4").output  ## (128 x 128)
    enc4 = vgg19.get_layer("block4_conv4").output  ## (64 x 64)

    """ Bridge """
    b1 = vgg19.get_layer("block5_conv4").output  ## (32 x 32)

    """ Decoder """
    dec1 = decoderBlock(b1, enc4, 512, 3)  ## (64 x 64)
    dec2 = decoderBlock(dec1, enc3, 256, 3)  ## (128 x 128)
    dec3 = decoderBlock(dec2, enc2, 128, 3)  ## (256 x 256)
    dec4 = decoderBlock(dec3, enc1, 64, 3)  ## (512 x 512)

    """ Output """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(dec4)

    model = Model(inputs, outputs, name="VGG19_U-Net")
    return model

def build_mobilenet(input_shape, classes, activation):
    backbone = MobileNetV2(input_shape=input_shape, weights=None, include_top=False)

    for i in range(0,len(backbone.layers)):
        #if "BatchNormalization" not in backbone.layers[i].name:
        backbone.layers[i].trainable = False

    x = Flatten()(backbone.output)

    # ultimo livello dense e softmax
    prediction = Dense(classes, activation=activation)(x) #softmax or sigmoid

    # creo il modello
    model = Model(inputs=backbone.input, outputs=prediction)

    return model

def freezeLayers(model, n_trainableLayers):
    # Freeze all Layers except the last n_trainbleLayers
    if n_trainableLayers > 0:
        lowerLim = len(model.layers) - n_trainableLayers
        for i in range(0, lowerLim):
            model.layers[i].trainable = False
        for i in range(lowerLim, len(model.layers)):
            model.layers[i].trainable = True
    # Freeze all Layers
    elif n_trainableLayers == 0:
        for i in range(0, len(model.layers)):
            model.layers[i].trainable = False

    # If n_trainableLayers < 0, do not freeze

    return model

def editClassifier(model, classes, activation, n_trainableLayers=-1):

    # Freeze all Layers except the last n_trainbleLayers
    model = freezeLayers(model, n_trainableLayers)

    x = Flatten()(model.output)

    # ultimo livello dense e softmax
    prediction = Dense(classes, activation=activation)(x) #softmax or sigmoid

    # creo il modello
    model = Model(inputs=model.input, outputs=prediction)

    return model


def build_vgg16_classification(input_shape, classes):
    model = Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=classes, activation="softmax"))

    return model

def build_vgg16_classification(input_shape, classes):
    model = Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # per VGG19 model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=classes, activation="softmax"))

    return model

def build_dummy(shape, n_classes):
    model = Sequential()
    model.add(InputLayer(input_shape=shape))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='sigmoid'))
    return model