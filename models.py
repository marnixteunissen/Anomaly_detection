import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot
import data_processing
import os


def build_conv_network(num_layers, filters, kernel=3, classes=2, activation='relu'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    model.add(layers.experimental.preprocessing.Rescaling(1. / 255))
    # construct network:
    for i in range(num_layers):
        model.add(layers.Conv2D(filters, kernel, activation=activation))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(classes))

    return model



model1 = keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)])

model1.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model1.fit(
  train_set,
  validation_data=val_set,
  epochs=1
)