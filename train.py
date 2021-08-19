import tensorflow as tf
import numpy as np
import pandas as pd
import models
import data_processing
import create_dataset
from file_functions import check_config
import os
import matplotlib.pyplot as plt
import json
from time import time
from datetime import datetime
import argparse
import sys


def save_losses(history, save_path):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(save_path)
    plt.clf()


def create_datasets(data_dir, img_size, batch_size, channel):
    train_data, val_data = data_processing.create_data_sets(data_dir, channel, 'train', batch_size, image_size=img_size)
    num_classes = len(train_data.class_names)
    assert len(train_data.class_names) == len(val_data.class_names)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)
    return train_data, val_data, num_classes


def train_network(conf_file=None):
    # opening the configuration file, and storing it in the destination directory
    if conf_file is None:
        # if no file is specified, the config file in this repository will be used
        f = "config.json"
    else:
        f = conf_file

    # Opening the configuration form the config file:
    file = open(f)
    config = json.load(file)
    print("Opened configuration from {}".format(f))

    # Checking if config file was filled out correctly:
    check_config(config)

    out_dir = config["output directory"]
    yes = {'yes', 'y', 'ye', ''}
    if out_dir == "":
        raise ValueError("Output path not specified")
    elif not os.path.exists(out_dir):
        choice = input("Output path does not exist, "
                       "would you like to create the new directory '{}'?".format(out_dir)).lower()
        if choice in yes:
            os.makedirs(out_dir)
        else:
            raise Exception('Output directory was not found, and not created, training aborted')

    # Parsing Configuration
    model_name = config["model name"]
    channels = config["video channels"]
    num_layers = config["number of layers"]
    num_filters = config["number of filters"]
    kernel = config["kernel size"]
    batch_size = config["batch size"]
    epochs = config['epochs']
    img_size = config["image size"]
    skip = config['maxpool frequency']

    # Check if path to dataset is set:
    if not os.path.exists(config["data directory"]):
        raise TypeError("Path to dataset not found")
    else:
        data_dir = config["data directory"]

    now = datetime.now().strftime("(%Y-%m-%d_%H.%M.%S)")
    model_paths = [os.path.join(out_dir, model_name + "_" + channel + now) for channel in channels]

    # Training is done per channel, one by one:
    for idx, channel in enumerate(channels):
        # Setting and checking output parameters:
        full_model_path = model_paths[idx]
        os.makedirs(full_model_path)
        conf_path = os.path.join(full_model_path, "config.json")
        results_path = os.path.join(full_model_path, "results.json")

        # Create the training and validation datasets:
        print(f"Creating datasets for {channel} channel")
        train_data, val_data, num_classes = create_datasets(data_dir, img_size, batch_size, channel)

        print(f"Creating test-dataset for {channel} channel")
        test_data = data_processing.create_test_set(data_dir, channel, img_size)
        print("")

        # Saving more configuration parameters
        config['trained channel'] = channel
        config['number of classes'] = num_classes
        with open(conf_path, "w") as fp:
            json.dump(config, fp, indent="")
            if not os.path.exists(conf_path):
                raise FileExistsError("Config file was not created, check writing privileges")

        # Creating the model:
        # Changes in the architecture should be implemented in the models.py file
        # TODO: have net_architecture option determine what type of model is built
        model = models.build_deep_CNN(num_layers, num_filters, img_size, kernel, num_classes, skip)

        # Save model structure summary:
        original_stdout = sys.stdout
        with open(os.path.join(full_model_path, "model_summary.txt"), "w") as f:
            sys.stdout = f
            model.summary()
            sys.stdout = original_stdout

        # Defining callbacks for training:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(full_model_path, 'saved_model', 'best', 'best_model.h5'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=10)

        # Start training:
        start_time = time()
        history = model.fit(train_data, validation_data=val_data, epochs=epochs, callbacks=[save_best, early_stop])
        finish_time = time()
        print(f"Total training time: {finish_time - start_time}")

        # Saving the trained model:
        model.save(os.path.join(full_model_path, 'saved_model', 'last_model.h5'))
        save_losses(history, (full_model_path + '/losses.png'))
        test_results = model.evaluate(test_data, return_dict=True)

        # Saving the training results
        results = {'training time': finish_time - start_time,
                   'number of epochs completed': len(history.history["loss"]), 'training results': history.history,
                   'test results after training': test_results}
        with open(results_path, "w") as fp:
            json.dump(results, fp, indent="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.json",
                        help='Full path to the config file, '
                             'if case none is specified the config file in hte root directory is used',
                        required=False)
    opt = parser.parse_args()
    train_network(opt.config)
