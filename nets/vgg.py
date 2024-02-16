import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input

def vgg_16(inputs):
    # Initialiser le modèle
    model = inputs
    # Ajouter les couches de convolution et de pooling
    model = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(model)
    model = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(model)

    model = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(model)
    model = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(model)

    model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(model)
    model = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(model)

    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(model)

    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(model)
    model = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(model)
    model = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(model)

    return model

