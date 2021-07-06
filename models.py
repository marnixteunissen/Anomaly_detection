import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, Softmax, \
    Dropout, ReLU, Concatenate, Conv2DTranspose, Input, Add, AveragePooling2D, ZeroPadding2D
import tensorflow.keras.losses as losses
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot
import data_processing
import os


def build_conv_network(num_layers, filters, image_size=(640, 360), kernel=3, classes=2, activation='relu', optimizer='adam'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    # model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.BatchNormalization(input_shape=(image_size[1], image_size[0], 3)))
    # construct network:
    for i in range(num_layers):
        if i % kernel == 0:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.MaxPooling2D())
        else:
            model.add(layers.Conv2D(filters, kernel, activation=activation))
            model.add(layers.MaxPooling2D(strides=(1, 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activation))
    model.add(layers.Dense(classes))
    model.add(layers.Softmax())
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


def VGG_like_network(num_layers, filters, image_size=(640, 360), kernel=5, classes=2, activation='relu', optimizer='adam'):
    # initialize model:
    model = keras.Sequential()
    # Normalising layer:
    # model.add(layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(image_size[0], image_size[1], 3)))
    model.add(layers.BatchNormalization(input_shape=(image_size[1], image_size[0], 3)))
    # construct network:
    for i in range(num_layers):
        model.add(layers.Conv2D(filters[i], kernel))
        model.add(layers.Conv2D(filters[i], kernel-2))
        model.add(layers.MaxPooling2D())

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation=activation))
    model.add(layers.Dense(classes))
    model.add(layers.Softmax())
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    return model


class ResNet():
    def __init__(self, num_blocks, num_layers, num_filters=[32, 64], input_shape=[640, 360, 3], activation='relu',
                 num_classes=2, optimizer='adam'):
        self.activation = activation
        self.num_filters = num_filters
        print(self.num_filters)
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.optim = optimizer

    def res_conv(self, x, s, filters):
        '''
        here the input size changes'''
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        # third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)

        # shortcut
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation(self.activation)(x)

        return x

    def res_identity(self, x, filters):
        # ReNet block where dimension does not change.
        # The skip connection is just simple identity connection
        # we will have 3 blocks and then input will be added

        x_skip = x # this will be used for addition with the residual block
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        # second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        # x = Activation(activations.relu)(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation(self.activation)(x)

        return x

    def build_model(self):
        # First stage:
        input_im = Input(shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        x = ZeroPadding2D(padding=(3, 3))(input_im)
        x = Conv2D(self.num_filters[0], kernel_size=7, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(self.activation)(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        num_filters = self.num_filters
        num_layers = self.num_layers

        # Create blocks
        for block in range(self.num_blocks-1):
            x = self.res_conv(x, s=1, filters=num_filters)
            for layer in range(num_layers):
                x = self.res_identity(x, filters=num_filters)
            num_filters = [f*2 for f in num_filters]
            num_layers += 1
        # Last layers:
        x = AveragePooling2D((2, 2), padding='same')(x)
        x = Flatten()(x)
        x = Dropout(0.1)(x)
        x = Dense(self.num_classes, kernel_initializer='he_normal')(x)
        x = Softmax()(x)
        # Define the model:
        model = Model(inputs=input_im, outputs=x, name='Resnet')
        model.compile(loss=losses.SparseCategoricalCrossentropy(),
                      optimizer=self.optim,
                      metrics=['accuracy'])
        return model


if __name__ == "__main__":
    data_dir = r'E:\Anomaly_detection'
    train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train', batch_size=8)
    num_classes = len(train_set.class_names)
    input_shape = [360, 640, 3]
    resnet = ResNet(2, 2)
    model = resnet.build_model()
    model.summary()
    # model = build_conv_network(2, 16)
    # model.summary()
    # model = VGG_like_network(5, [16, 32, 64, 128, 256])
    print(train_set)
    history = model.fit(train_set, validation_data=val_set, epochs=1)
