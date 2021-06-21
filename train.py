import tensorflow as tf
from itertools import cycle
import numpy as np
import pandas as pd
import models
import data_processing
import create_dataset
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


def run_layer_filter_experiments(layers, filters, image_size, batch_size, kernels, data_dir=None, out_dir=os.getcwd(), epochs=3):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data\data-set'
    train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size)
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
        train_ds = []
        val_ds = []
        n_layers = []
        n_filters = []
        image_size = []
        batch_size = []
        optimizer = 'adam'
        epochs = []
        kernel = []


    @ex.capture
    def data(image_size, batch_size):
        train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size, image_size=image_size)
        return train_data, val_data

    @ex.capture
    def build_model(n_layers, filters, image_size, kernel):
        model = models.build_conv_network(n_layers, filters, kernel=kernel, image_size=image_size)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # add early stopping criterion:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
        # start training:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save_best, early_stop])
        model.save(os.path.join(ex.observers[0].dir, 'saved_model/'))

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

    @ex.capture
    def test(model, image_size):
        test_set = data_processing.create_test_set(data_dir, channel='TOP', image_size=image_size)
        result = model.evaluate(test_set)
        return result

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

        # run test
        test(model)

    i = 1
    experiments = {}
    experiment_list = list(itertools.product(layers, filters, kernels))
    experiments['layers'] = layers
    experiments['filters'] = filters
    experiments['kernels'] = kernels


    results = {}

    for l, f, k in zip(experiments['layers'], experiments['filters'], experiments['kernels']):
        print('layers: {}, filters: {}'.format(l, f))
        conf = {'n_layers': int(l),
                'filters': f,
                'image_size': image_size,
                'batch_size': batch_size,
                'epochs': epochs,
                'kernel': k}

        exp_finish = ex.run(config_updates=conf)

        results['layers: {}, filters: {}'.format(l, f)] = exp_finish.result
        # print('run {}/{} complete'.format(i,len(experiments)))

        i = i+1

def run_VGG_experiments(layers, filters, image_size, batch_size, kernels, data_dir=None, out_dir=os.getcwd(), optimizer='adam', epochs=3):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data\data-set'
    train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size)
    run_path = os.path.join(out_dir, 'runs')

    ex = Experiment('VGG architectures')
    ex.observers.append(FileStorageObserver(basedir=os.path.join(run_path, ex.path)))
    # loss_path = os.path.join(run_path, ex.path, 'losses.png')

    @ex.config
    def config():
        """This is the configuration of the experiment"""
        dataset_name = 'Anomaly Detection'
        net_architecture = 'VGG like'
        data_dir = os.getcwd() + r'\data\data-set'
        train_ds = []
        val_ds = []
        n_layers = []
        n_filters = []
        image_size = []
        batch_size = []
        optimizer = 'adam'
        epochs = []

    @ex.capture
    def data(image_size, batch_size):
        train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size, image_size=image_size)
        return train_data, val_data

    @ex.capture
    def build_model(n_layers, filters, image_size):
        model = models.VGG_like_network(n_layers, filters, image_size=image_size)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # create callback for early stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=3)
        # start training:
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[save_best, early_stop])
        model.save(os.path.join(ex.observers[0].dir, 'saved_model/'))

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

    @ex.capture
    def test(model, image_size):
        test_set = data_processing.create_test_set(data_dir, channel='TOP', image_size=image_size)
        result = model.evaluate(test_set)
        return result

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

        # run test
        test(model)

    i = 1
    experiments = {}
    experiments['layers'] = layers
    experiments['filters'] = filters
    experiments['kernels'] = kernels


    results = {}

    for l, f, k in zip(experiments['layers'], experiments['filters'], experiments['kernels']):
        print('layers: {}, filters: {}'.format(l, f))
        conf = {'n_layers': int(l),
                'filters': f,
                'image_size': image_size,
                'batch_size': batch_size,
                'epochs': epochs,
                'kernel': k}

        exp_finish = ex.run(config_updates=conf)

        results['layers: {}, filters: {}'.format(l, f)] = exp_finish.result
        # print('run {}/{} complete'.format(i,len(experiments)))

        i = i+1


if __name__ == "__main__":
    layersvgg = [5, 6]
    filtersvgg = [[64, 128, 256, 512, 512], [64, 128, 256, 512, 512]]
    layers = [10]
    filters = [64]
    kernels = [5, 7]
    CNN = True
    VGG = True

    if CNN:
        run_layer_filter_experiments(layers, filters, kernels=kernels, image_size=(640, 360), batch_size=32,
                                     data_dir=r'E:\Anomaly_detection', epochs=2)
    if VGG:
        run_VGG_experiments(layersvgg, filtersvgg, (640, 360), batch_size=8,
                            data_dir=r'E:\Anomaly_detection', epochs=10)
