import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import cv2
import models
import csv
import data_processing
from create_dataset import delays as Delays
import excel_functions as ex


def draw_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 320)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    return cv2.putText(img, text, org, font, fontScale, fontColor, thickness, cv2.LINE_AA)


def detect_video(source, model, image_size, conf=0.70):
    class_names = {0: "FJOK", 1: "NONE"}
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Video not found")
        exit()
    prob = {'FJOK': [], 'NONE': []}
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        model_input = tf.expand_dims(cv2.resize(frame, image_size), 0)

        # Run inference on frame:
        pred = model(model_input)
        frame_prob = tf.nn.softmax(pred).numpy()[0]
        prob['FJOK'].append(frame_prob[0])
        prob['NONE'].append(frame_prob[1])
        if frame_prob[0] >= conf:
            label = 'FJOK'
        elif frame_prob[1] >= conf:
            label = 'NONE'
        else:
            label = ''
        # label = class_names[np.argmax(frame_prob)]

        frame = draw_label(frame, label)
        cv2.imshow('Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    plt.figure()
    plt.title("Probabilities")
    plt.plot(range(len(prob['FJOK'])), prob['FJOK'], label="Field Joint")
    plt.plot(range(len(prob['NONE'])), prob['NONE'], label="None")
    plt.xlabel("Frame")
    plt.ylabel("class probability")
    plt.legend()
    plt.show()


def plot_detect_video(source, project, model, image_size, dir):
    class_names = {0: "FJOK", 1: "NONE"}
    delays = Delays()
    offset = - delays[project]
    excel = ex.extract_excel_data(r'data/'+project)
    vid_events = ex.extract_video_events(excel, source, offset)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not os.path.exists(dir):
        os.makedirs(dir)

    if not cap.isOpened():
        print("Video not found")
        exit()
    prob = {'FJOK': [], 'NONE': []}
    if 'prediction.csv' not in os.listdir(dir):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            model_input = tf.expand_dims(cv2.resize(frame, image_size), 0)

            # Run inference on frame:
            pred = model(model_input)
            frame_prob = tf.nn.softmax(pred).numpy()[0]
            prob['FJOK'].append(frame_prob[0])
            prob['NONE'].append(frame_prob[1])
        df = pd.DataFrame(prob)
        df.to_csv(dir + '/prediction.csv')
    else:
        df = pd.read_csv(dir + '/prediction.csv')

    df['rolling'] = df.FJOK.rolling(60).mean()
    print(vid_events['ms in video'])

    plt.figure()
    plt.title("Probabilities")
    # plt.plot(range(len(df['FJOK'])), df['FJOK'], label="Field Joint")
    plt.plot(range(len(df['rolling'])), df['rolling'], label='Rolling Average')
    plt.vlines((vid_events['ms in video']*0.03).round().astype(int), 0, 1, linestyles='dashed', colors='r')
    plt.xlabel("Frame")
    plt.ylabel("class probability")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    dir = os.getcwd()
    class_names = {0: "FJOK", 1: "NONE"}
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/62')

    with open(exp_dir + r'/config.json') as f:
        img_size = tuple(json.load(f)['image_size']['py/tuple'])

    project = 'Troll'
    model = tf.keras.models.load_model(exp_dir+'/saved_model')
    video = os.path.join(dir, 'data', 'video', '20200423213211791@MainDVR_Ch2_Trim.mp4')
    plot_detect_video(video, project, model, img_size, r'data/video/predictions_3')

