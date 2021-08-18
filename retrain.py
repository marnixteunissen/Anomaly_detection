import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Input, Model, losses, optimizers
from os import listdir, makedirs
from os.path import join
import argparse
import data_processing as data
from time import time
from datetime import datetime
import train
from shutil import copy


def retrain(model_dir, data_dir='', batch_size=32, epochs=50):
    # TODO: get config from file
    img_size = (360, 640)
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ret_dir = model_dir + '/retrain/' + now
    makedirs(ret_dir)
    conf = model_dir + '/config.json'
    conf_dest = ret_dir + '/config.json'
    copy(conf, conf_dest)

    # create new dataset
    train_set, val_set, num_classes = train.create_datasets(data_dir, img_size, batch_size)
    test_set = data.create_test_set(data_dir, 'TOP', img_size)

    # Import pre_trained model
    print("Loading Model...", flush=True)
    if 'best' in listdir(model_dir + '/saved_model'):
        if 'best_model.h5' in listdir(model_dir + '/saved_model/best'):
            old_model = load_model(model_dir + '/saved_model/best/best_model.h5')
        else:
            old_model = load_model(model_dir + '/saved_model/best')
    else:
        if 'saved_model.h5' in listdir(model_dir + '/saved_model'):
            old_model = load_model(model_dir + '/saved_model.h5')
        else:
            old_model = load_model(model_dir + '/saved_model')

    base_model = tf.keras.models.Model(inputs=old_model.inputs, outputs=old_model.layers[-4].get_output_at(0))
    # Freeze layers
    base_model.trainable = False

    # Create new top layers
    dense_layer = layers.Dense(128, activation='relu')
    prediction_layer = layers.Dense(num_classes, activation='relu')
    outputs = layers.Softmax()

    # Create new model
    inputs = Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = dense_layer(x)
    x = prediction_layer(x)
    output = outputs(x)

    new_model = Model(inputs, output)
    optimizer = optimizers.Adam(amsgrad=True)
    new_model.compile(optimizer=optimizer,
                      loss=losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
    new_model.summary()

    # setup for training:
    # create callback to save best model:
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(ret_dir, 'saved_model', 'best', 'best_model.h5'),
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # add early stopping criterion:
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5)

    # Start training
    start_time = time()
    history = new_model.fit(train_set, validation_data=val_set, epochs=epochs, callbacks=[save_best, early_stop])
    finish_time = time()
    print("Total training time: {}".format(finish_time-start_time))
    new_model.save(join(ret_dir, 'saved_model', 'last_model.h5'))
    train.save_losses(history, (ret_dir + '/losses.png'))
    new_model.evaluate(test_set)


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None,
                        help='Full path to the data directory',
                        required=False)
    parser.add_argument('--model', type=str, default=None,
                        help="Full path to the directory which contains the 'saved_model' directory",
                        required=False)
    opt = parser.parse_args()
    retrain(opt.model, opt.data, batch_size=16)
