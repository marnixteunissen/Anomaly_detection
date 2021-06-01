import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow.keras.losses as losses
import numpy as np
import matplotlib.pyplot
import data_processing
import os


def build_conv_network(num_layers, filters, image_size=(640, 360), kernel=3, classes=2, activation='relu', optimizer='adam'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    # model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.BatchNormalization(input_shape=(image_size[0], image_size[1], 3)))
    # construct network:
    for i in range(num_layers):
        if i % 2 == 0:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.MaxPooling2D())
        else:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.MaxPooling2D(strides=(1, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(classes))

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model


def VGG_like_network(num_layers, filters, image_size=(640, 360), kernel=3, classes=2, activation='relu', optimizer='adam'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    # model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.BatchNormalization(input_shape=(image_size[0], image_size[1], 3)))
    # construct network:
    for i in range(num_layers):
        model.add(layers.Conv2D(filters[i], kernel))
        model.add(layers.Conv2D(filters[i], kernel))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation=activation))
    model.add(layers.Dense(classes))

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    data_dir = r'E:\Anomaly_detection'
    train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size=8)
    num_classes = len(train_set.class_names)

    model = VGG_like_network(5, [16, 32, 64, 128, 256])

    model.summary()

    history = model.fit(train_set, validation_data=val_set, epochs=1)
