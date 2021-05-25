import os
import tensorflow as tf
import numpy as np
import json
import cv2
import models
import data_processing

def draw_label(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 320)
    fontScale = 1
    fontColor = (255, 255, 255)
    thickness = 2
    return cv2.putText(img, text, org, font, fontScale, fontColor, thickness, cv2.LINE_AA)

if __name__ == "__main__":
    class_names = {0: "FJOK", 1: "NONE"}
    exp_dir = os.path.abspath(r'runs/Varying layers and filters/25')
    # f = open((exp_dir+'/config.json'))
    # exp_para = json.load(f)
    # filters = exp_para['n_filters']
    # layers = exp_para['n_layers']
    model = tf.keras.models.load_model(exp_dir+'/saved_model')
    model.summary()


    # model = models.build_conv_network(layers, filters)
    # model.load_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\runs\Varying layers and filters\20\weights')

    # model.save_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\weights')
    # model.load_weights(r'C:\Users\MTN\Documents\Survey_anomaly_detection\pycharm\Anomaly_detection\weights')
    # model.summary()
    # model.load_weights((exp_dir + r'/best_weights.ckpt'))

    cap = cv2.VideoCapture('20200423173206224@MainDVR_Ch2.mp4')
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        model_input = tf.expand_dims(cv2.resize(frame, (480, 360)), 0)
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Run inference on frame:
        pred = model(model_input)
        print(pred.numpy())
        prob = tf.nn.softmax(pred)
        print('probabilities: ', prob)
        pred = np.argmax(pred.numpy())
        label = class_names[pred]
        frame = draw_label(frame, (label+'({}%)'.format(prob*100)))
        # Display the resulting frame
        cv2.imshow('Feed', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
