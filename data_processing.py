import numpy as np
import os
import PIL
import PIL.Image
import tensorflow.keras.preprocessing as preprocessing
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def create_data_sets(data_dir, channel, mode, batch_size=32, image_size=[480, 360]):
    data_dir = os.path.join(data_dir, channel, mode)
    width, height = image_size[0], image_size[1]
    train_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)
    val_ds = preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    return train_ds, val_ds


if __name__ == "__main__":
    train, val = create_data_sets(r'data\data-set', 'TOP', 'train', batch_size=4)
    class_names = train.class_names
    print(class_names)
    norm_layer = layers.experimental.preprocessing.Rescaling(1./255)
