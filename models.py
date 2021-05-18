import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot
import data_processing
import os


def build_conv_network(num_layers, filters, kernel=3, classes=2, activation='relu', optimizer='adam'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(360, 480, 3)))
    # construct network:
    for i in range(num_layers):
        model.add(layers.Conv2D(filters, kernel, activation=activation))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(classes))

    model.compile(
        optimizer=optimizer,
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

    return model

if __name__ == "__main__":
    data_dir = os.getcwd() + r'\data\data-set'
    train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train')
    num_classes = len(train_set.class_names)

    model = build_conv_network(3, 16)

    model.summary()

    history = model.fit(train_set, validation_data=val_set, epochs=1)