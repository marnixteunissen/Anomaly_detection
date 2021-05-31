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


def eval_show_false(model, data_dir,save_dir , save_false=False, channel='TOP'):
    class_names = {0: "FJOK", 1: "NONE"}
    test_set = create_test_set(data_dir, channel)
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

            # cv2.imshow('Class: {}, Prediction: {} ({})'.format(class_names[label],
            #                                                    class_names[pred],
            #                                                    prob.numpy()[0][pred]),
            #            image.numpy().astype("uint8"))
            if save_false:
                if not os.path.exists(save_dir+r'/wrong_labels'):
                    os.mkdir(save_dir+r'/wrong_labels')
                cv2.imwrite(save_dir+r'/wrong_labels/{}_{}.png'.format(class_names[pred], prob.numpy()[0][pred]),
                            image.numpy().astype("uint8"))
            # if cv2.waitKey(1) == ord('q'):
            #     continue


if __name__ == "__main__":
    data_dir = r'E:\Anomaly_detection'
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/42')
    model_load = tf.keras.models.load_model(exp_dir+'/saved_model')
    evaluate_network(model_load, data_dir)
    #eval_show_false(model_load, data_dir, save_dir=exp_dir, save_false=True)
