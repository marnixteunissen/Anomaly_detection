import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, Input, Model
from os import listdir
from os.path import join
import argparse
import excel_functions as ex
import data_processing as data
import train


def retrain(model_dir, data_dir='', save_dir='', batch_size=32, epochs=30):
    # TODO: get config from file
    classes = 2
    img_size = (360, 640)

    # create new dataset
    train_set, val_set = train.create_datasets(data_dir, img_size, batch_size)
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

    base_model = tf.keras.models.Model(inputs=old_model.inputs, outputs=old_model.layers[-5].get_output_at(0))
    # Freeze layers
    base_model.trainable = False

    # Create new top layers
    pool_layer = layers.GlobalAvgPool2D()
    dense_layer = layers.Dense(128, activation='relu')
    prediction_layer = layers.Dense(classes, activation='relu')
    outputs = layers.Softmax()

    # Create new model
    inputs = Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = pool_layer(x)
    x = dense_layer(x)
    x = prediction_layer(x)
    output = outputs(x)

    new_model = Model(inputs, output)
    new_model.summary()

    # setup for training:
    # create callback to save best model:
    save_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=join(save_dir, 'saved_model/best/best_model.h5'),
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    # add early stopping criterion:
    early_stop = tf.keras.callbacks.EarlyStopping(patience=8)

    # Start training
    history = new_model.fit(train_set, validation_data=val_set, epochs=epochs, callbacks=[save_best, early_stop])
    new_model.save(join(save_dir, 'saved_model/last_model.h5'))
    train.save_losses(history, save_dir)
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
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Full path to location to store the new model')
    opt = parser.parse_args()
    path = r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results\29-07-2021\Varying layers and filters\1'
    retrain(path)
