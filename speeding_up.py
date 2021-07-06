import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from time import time


def change_model_format(model_dir):
    # Load the model
    start = time()
    if 'best' in os.listdir(model_dir + '/saved_model'):
        model = load_model(model_dir + '/saved_model/best')
    else:
        model = load_model(model_dir + '/saved_model')
    stop = time()
    print(f"Loading time model: {stop-start} seconds")




if __name__=="__main__":
    model_source = r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\runs\ResNet architectures\8'
    change_model_format(model_source)