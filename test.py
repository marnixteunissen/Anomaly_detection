import tensorflow as tf
from data_processing import create_test_set
import json
from models import build_conv_network
import data_processing
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


def evaluate_network(model, data_dir, image_size, channel='TOP'):
    test_set = create_test_set(data_dir, channel, image_size=image_size)
    result = model.evaluate(test_set)


def eval_show_false(model, data_dir, image_size, save_dir, save_false=False, channel='TOP'):
    class_names = {0: "FJOK", 1: "NONE"}
    test_set = create_test_set(data_dir, channel, image_size=image_size)
    # result = model.evaluate(test_set)

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
            if save_false:
                if not os.path.exists(save_dir+r'/wrong_labels'):
                    os.mkdir(save_dir+r'/wrong_labels')
                cv2.imwrite(save_dir+r'/wrong_labels/{}_{}.png'.format(class_names[pred], prob.numpy()[0][pred]),
                            image.numpy().astype("uint8"))
            if cv2.waitKey(1) == ord('q'):
                continue


def eval_save_false(model, data_dir, image_size, save_dir, channel='TOP'):
    class_names = {0: "FJOK", 1: "NONE"}
    test_set = create_test_set(data_dir, channel, image_size=image_size)

    for image, label in test_set:
        label = label.numpy()[0]
        pred = model(image)
        prob = tf.nn.softmax(pred)
        pred = np.argmax(model(image).numpy())

        if pred != label:
            image = tf.squeeze(image)
            if not os.path.exists(save_dir+r'/wrong_labels'):
                os.mkdir(save_dir+r'/wrong_labels')
            cv2.imwrite(save_dir+r'/wrong_labels/{}_{}.png'.format(class_names[pred], prob.numpy()[0][pred]),
                        image.numpy().astype("uint8"))


if __name__ == "__main__":
    data_dir = r'E:\Anomaly_detection'
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/121')

    with open(exp_dir+r'/config.json') as f:
        img_size = json.load(f)['image_size']['py/tuple']
    print(img_size)

    model_load = tf.keras.models.load_model(exp_dir+'/saved_model')
    evaluate_network(model_load, data_dir, img_size)
    # eval_show_false(model_load, data_dir, img_size, save_dir=exp_dir, save_false=True)
    eval_save_false(model_load, data_dir, img_size, save_dir=exp_dir)
