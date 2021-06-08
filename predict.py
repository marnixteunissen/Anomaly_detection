import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import models
import data_processing
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


def plot_detect_video(source, project, model, image_size):
    class_names = {0: "FJOK", 1: "NONE"}
    excel = ex.extract_excel_data(project)
    vid_events = ex.extract_video_events(excel,)
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


if __name__ == "__main__":
    dir = os.getcwd()
    class_names = {0: "FJOK", 1: "NONE"}
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/117')

    with open(exp_dir + r'/config.json') as f:
        img_size = tuple(json.load(f)['image_size']['py/tuple'])

    model = tf.keras.models.load_model(exp_dir+'/saved_model')
    video = os.path.join(dir, 'data', 'video', '20200423160203385@MainDVR_Ch2_Trim.mp4')
    detect_video(video, model, img_size)

    # model = models.build_conv_network(layers, filters)
    # model.load_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\runs\Varying layers and filters\20\weights')

    # model.save_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\weights')
    # model.load_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\weights')
    # model.summary()
    # model.load_weights((exp_dir + r'/best_weights.ckpt'))

    # cap = cv2.VideoCapture('20200423173206224@MainDVR_Ch2.mp4')
    # if not cap.isOpened():
    #     print("Cannot open camera")
    #     exit()
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #     model_input = tf.expand_dims(cv2.resize(frame, (480, 360)), 0)
    #     # if frame is read correctly ret is True
    #     if not ret:
    #         print("Can't receive frame (stream end?). Exiting ...")
    #         break
    #     # Run inference on frame:
    #     pred = model(model_input)
    #     print(pred.numpy())
    #     prob = tf.nn.softmax(pred)
    #     print('probabilities: ', prob)
    #     pred = np.argmax(pred.numpy())
    #     label = class_names[pred]
    #     frame = draw_label(frame, (label+'({}%)'.format(prob*100)))
    #     # Display the resulting frame
    #     cv2.imshow('Feed', frame)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
    # # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()
