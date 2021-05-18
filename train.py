import tensorflow as tf
import numpy as np
import pandas as pd
import models
import data_processing
import os
import itertools
from sacred import Experiment
from sacred.observers import FileStorageObserver
import matplotlib.pyplot as plt

# train models with varying nr of layers and filters:
# logging done with sacred: https://github.com/sakoarts/sacred_presentation/blob/master/sacred_presentation.ipynb


def train_experiment(model, train_ds, val_ds, epochs):
    model.summary()
    print("Starting model training...")
    # start training:
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    return history


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


def run_layer_filter_experiments(layers, filters, out_dir=os.getcwd(), optimizer='adam', epochs=3):


    data_dir = os.getcwd() + r'\data\data-set'
    train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train')
    run_path = os.path.join(out_dir, 'runs')

    ex = Experiment('Varying layers and filters')
    ex.observers.append(FileStorageObserver(basedir=os.path.join(run_path, ex.path)))
    # loss_path = os.path.join(run_path, ex.path, 'losses.png')

    @ex.config
    def config():
        """This is the configuration of the experiment"""
        dataset_name = 'Anomaly Detection'
        net_architecture = 'Simple CNN'
        data_dir = os.getcwd() + r'\data\data-set'
        # train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train')

        # run_path = os.path.join(out_dir, 'runs')

        ## number of layers
        # n_layers = []
        ## number of filters per layer
        # n_filters = []
        ## Optimizer:
        # optim = []
        ## Epochs:
        # ep = []

    @ex.capture
    def data():
        train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train')
        return train_data, val_data

    @ex.capture
    def build_model(n_layers, n_filters, optimizer):
        model = models.build_conv_network(n_layers, n_filters, optimizer=optimizer)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # model.summary()
        # start training:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

        return history

    @ex.capture
    def save_losses(history):
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper right')
        plt.savefig(os.path.join(ex.observers[0].dir, 'losses.png'))
        plt.clf()

    @ex.main
    def main():
        # Get data
        print('Creating data-sets...')
        train_ds, val_ds = data()

        # build network
        print('Building model...')
        model = build_model()

        # train network
        print('Training network:')
        history = train(model, train_ds, val_ds)

        # Save plot with losses
        print('Saving losses...')
        save_losses(history)

        print('Final accuracy: ', history.history['val_accuracy'][-1])


    i = 1
    experiments = list(itertools.product(layers, filters))

    for l, f in experiments:
        print('layers: {}, filters: {}'.format(l, f))
        conf = {'train_ds': train_data,
                'val_ds': val_data,
                'n_layers': int(l),
                'n_filters': int(f),
                'optimizer': optimizer,
                'epochs': epochs,
                }
        exp_finish = ex.run(config_updates={'n_layers': l,
                                            'n_filters': f,
                                            'optimizer': optimizer,
                                            'epochs': epochs})

        print('run {}/{} complete'.format(i,len(experiments)))

        i = i+1


if __name__ == "__main__":
    layers = [2, 4, 6]
    filters = [8, 16, 32]

    run_layer_filter_experiments(layers, filters, epochs=1)
