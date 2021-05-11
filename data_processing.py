import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf


def create_data_sets(data_dir, channel, mode, batch_size=32, image_size=[480, 360]):
    data_dir = os.path.join(data_dir, channel, mode)
    width, height = image_size[0], image_size[1]
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(height, width),
        batch_size=batch_size)

    return train_ds, val_ds
