import tensorflow as tf
import tensorflow.keras.losses as losses
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


def create_datasets(data_dir, img_size, batch_size):
    train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size, image_size=img_size)
    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.AUTOTUNE)
    return train_data, val_data


def run_layer_filter_experiments(layers, filters, image_size, batch_size, kernels, data_dir=None, out_dir=os.getcwd(), epochs=3):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data\data-set'
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
    def build_model(n_layers, filters, image_size, kernel):
        model = models.build_conv_network(n_layers, filters, kernel=kernel, image_size=image_size)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # add early stopping criterion:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=8)
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
        train_ds, val_ds = create_datasets(data_dir, image_size, batch_size)

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


def run_VGG_experiments(layers, filters, image_size, batch_size, kernels, data_dir=None, out_dir=os.getcwd(),
                        optimizer='adam', epochs=3):
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
    def build_model(n_layers, filters, kernel, image_size):
        model = models.VGG_like_network(n_layers, filters, image_size=image_size, kernel=kernel)
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # create callback for early stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=8)
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


def run_ResNet_experiments(blocks, layers, filters, image_size, batch_size, kernels=3, data_dir=None, out_dir=os.getcwd(),
                        optimizer='adam', epochs=3):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data\data-set'
    train_data, val_data = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size)
    run_path = os.path.join(out_dir, 'runs')

    ex = Experiment('ResNet architectures')
    ex.observers.append(FileStorageObserver(basedir=os.path.join(run_path, ex.path)))
    # loss_path = os.path.join(run_path, ex.path, 'losses.png')

    @ex.config
    def config():
        """This is the configuration of the experiment"""
        dataset_name = 'Anomaly Detection'
        net_architecture = 'ResNet'
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
    def build_model(n_blocks, n_layers, filters, image_size):
        resnet = models.ResNet(n_blocks, n_layers, filters)
        model = resnet.build_model()
        model.summary()
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # create callback for early stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=8)
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
        train_ds, val_ds = create_datasets(data_dir, image_size, batch_size)

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
    experiments['blocks'] = blocks


    results = {}

    for l, f, b in zip(experiments['layers'], experiments['filters'], experiments['blocks']):
        assert(len(f) == 2)
        print('blocks: {}, layers: {}, filters: {}'.format(b, l, f))
        conf = {'n_layers': int(l),
                'filters': f,
                'n_blocks': b,
                'image_size': image_size,
                'batch_size': batch_size,
                'epochs': epochs}

        exp_finish = ex.run(config_updates=conf)

        results['layers: {}, filters: {}'.format(l, f)] = exp_finish.result
        # print('run {}/{} complete'.format(i,len(experiments)))

        i = i+1


def run_precompiled_experiments(model_type, batch_size, weights='imagenet', image_size=[640, 360, 3], data_dir=None, out_dir=os.getcwd(),
                        optimizer='adam', epochs=3):
    if data_dir is None:
        data_dir = os.getcwd() + r'\data\data-set'
    run_path = os.path.join(out_dir, 'runs')

    ex = Experiment(model_type)
    ex.observers.append(FileStorageObserver(basedir=os.path.join(run_path, ex.path)))

    @ex.config
    def config():
        """This is the configuration of the experiment"""
        dataset_name = 'Anomaly Detection'
        net_architecture = []
        data_dir = os.getcwd() + r'\data\data-set'
        train_ds = []
        val_ds = []
        weights = []
        image_size = []
        batch_size = []
        optimizer = 'adam'
        epochs = []

    @ex.capture
    def build_model(net_architecture):
        if net_architecture == 'DenseNet121':
            pre_process = tf.keras.applications.densenet.preprocess_input
            base_model = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                                                    weights=weights,
                                                                    input_shape=image_size)

        elif net_architecture == 'DenseNet169':
            pre_process = tf.keras.applications.densenet.preprocess_input
            base_model = tf.keras.applications.densenet.DenseNet169(include_top=False,
                                                                    weights=weights,
                                                                    input_shape=image_size)

        elif net_architecture == 'MobileNetV2':
            pre_process = tf.keras.applications.mobilenet_v2.preprocess_input
            base_model = tf.keras.applications.MobileNetV2(input_shape=image_size,
                                                           include_top=False,
                                                           weights=weights)
        if weights is not None:
            base_model.trainable = False

        flatten = tf.keras.layers.Flatten()
        prediction_layer = tf.keras.layers.Dense(2)

        inputs = tf.keras.Input(shape=(image_size[0], image_size[1], image_size[2]))
        x = pre_process(inputs)
        if weights is not None:
            x = base_model(x, training=False)
        else:
            x = base_model(x)
        x = flatten(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = prediction_layer(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=optimizer, loss=losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        model.summary()
        return model

    @ex.capture
    def train(model, train_ds, val_ds, epochs):
        # create callback to save best model:
        save_best = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ex.observers[0].dir, 'saved_model/best'),
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # create callback for early stopping:
        early_stop = tf.keras.callbacks.EarlyStopping(patience=10)
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
        train_ds, val_ds = create_datasets(data_dir, image_size, batch_size)

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

    print()
    conf = {'image_size': image_size,
            'net_architecture': model_type,
            'weights': weights,
            'batch_size': batch_size,
            'epochs': epochs}

    exp_finish = ex.run(config_updates=conf)

    results = exp_finish.result


if __name__ == "__main__":
    # CNN parameters
    layers = [14, 14, 14, 14, 14, 12]
    filters = [32, 64, 128, 256, 32, 32]
    kernels = [3, 3, 3, 3, 5, 5]

    # VGG parameters
    filtersvgg = [[16, 16, 32, 64, 128], [16, 32, 64, 128, 128], [32, 64, 128, 256, 256], [64, 64, 64, 64, 64, 64, 64]]
    layersvgg = [len(f) for f in filtersvgg]

    # ResNet parameters
    num_blocks = [6, 6, 6, 6, 6, 6]
    num_layers = [2, 2, 2, 3, 3, 3]
    num_filters = [[8, 16], [16, 32], [32, 64], [8, 16], [16, 32], [32, 64]]

    CNN = True
    VGG = False
    ResNet = False
    Other = False
    epochs = 50
    batch_size = 16
    image_size = (360, 640)
    data_dir = r'E:\Anomaly_detection'
    out_dir = r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results'
    if ResNet:
        run_ResNet_experiments(num_blocks, num_layers, num_filters, image_size=(640, 360), batch_size=8,
                               data_dir=data_dir, epochs=epochs,
                               out_dir=out_dir)
    if CNN:
        run_layer_filter_experiments(layers, filters, kernels=kernels, image_size=(640, 360), batch_size=8,
                                     data_dir=data_dir, epochs=epochs,
                                     out_dir=out_dir)
    if VGG:
        run_VGG_experiments(layersvgg, filtersvgg, kernels=kernels, image_size=(640, 360), batch_size=8,
                            data_dir=data_dir, epochs=epochs,
                            out_dir=r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results')
    if ResNet:
        run_ResNet_experiments(num_blocks, num_layers, num_filters, image_size=(640, 360), batch_size=8,
                               data_dir=data_dir, epochs=20)
    if Other:
        for arch in ['DenseNet121', 'DenseNet169', 'MobileNetV2']:
            run_precompiled_experiments(arch, batch_size=8, data_dir=data_dir,
                                        optimizer='adam', epochs=epochs,
                                        out_dir=r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results')

        run_precompiled_experiments('MobileNetV2', weights=None, batch_size=8, data_dir=data_dir,
                                    optimizer='adam', epochs=epochs,
                                    out_dir=r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results')
        run_precompiled_experiments('MobileNetV2', weights='imagenet', batch_size=8, data_dir=data_dir,
                                    optimizer='adam', epochs=epochs,
                                    out_dir=r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results')
