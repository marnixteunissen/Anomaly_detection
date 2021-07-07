import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from time import time
import threading
from multi_thread_read import FileVideoStream
import models
import json


def build_resnet(model_dir):
    config_file = os.path.abspath(os.path.join(model_dir, 'config.json'))
    f = open(config_file)
    config = json.load(f)
    blocks, layers, filters = config['n_blocks'], config['n_layers'], config['filters']
    resnet = models.ResNet(blocks, layers, filters)
    model = resnet.build_model()
    return model

def change_model_format(model_dir):
    # Load the model
    start = time()
    if 'best' in os.listdir(model_dir + '/saved_model'):
        model = load_model(model_dir + '/saved_model/best')
        model.save(model_dir + '/saved_model/best/best_model.h5')
    else:
        model = load_model(model_dir + '/saved_model')
        model.save(model_dir + '/saved_model/saved_model.h5')
    stop = time()
    print(f"Loading time model: {stop-start} seconds")



if __name__=="__main__":
    video_file = r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\data\video\20200423213211791@MainDVR_Ch2_Trim3.mp4'
    model_source = r'K:\PROJECTS\SubSea Detection\10 - Development\Training Results\runs\VGG architectures\1'
    # fvs = FileVideoStream(video_file, model_source).start()

    change_model_format(model_source)
