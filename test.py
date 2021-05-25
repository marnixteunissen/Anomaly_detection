import tensorflow as tf
from data_processing import create_test_set
from models import build_conv_network
import data_processing
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def evaluate_network(model, data_dir, channel='TOP'):
    test_set = create_test_set(data_dir, channel)
    result = model.evaluate(test_set)


def eval_show_false(model, data_dir, channel='TOP'):
    class_names = {0: "FJOK", 1: "NONE"}
    test_set = create_test_set(data_dir, channel)
    result = model.evaluate(test_set)
    i = 1
    for image, label in test_set:
        label = label.numpy()[0]
        pred = model(image)
        prob = tf.nn.softmax(pred)
        pred = np.argmax(model(image).numpy())
        if pred != label:
            print('label:', label)
            print('Prediction: {} ({}%)'.format(pred, prob.numpy()[0][pred]*100))
            image = tf.squeeze(image)

            cv2.imshow('Class: {}, Prediction: {} ({})'.format(class_names[label],
                                                               class_names[pred],
                                                               prob.numpy()[0][pred]),
                       image.numpy().astype("uint8"))
            cv2.waitKey(2000)
        i = i+1
        if i == 10:
            cv2.waitKey(1)
            break



if __name__ == "__main__":
    data_dir = r'E:\Anomaly_detection'
    train_set, val_set = data_processing.create_data_sets(data_dir, 'TOP', 'train')

    test_model = build_conv_network(3, 16)
    # test_model.fit(train_set, validation_data=val_set, epochs=5)
    # evaluate_network(test_model, data_dir)
    # eval_show_false(test_model, data_dir)
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/25')
    model_load = tf.keras.models.load_model(exp_dir+'/saved_model')
    evaluate_network(model_load, data_dir)
