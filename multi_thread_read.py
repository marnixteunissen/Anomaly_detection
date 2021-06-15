from threading import Thread
import cv2
from queue import Queue
import time
import tensorflow as tf
import json
import pandas as pd
import os
from create_dataset import delays as Delays
import excel_functions as ex
import matplotlib.pyplot as plt
import datetime
import argparse


class FileVideoStream:
    def __init__(self, path, model_dir, queueSize=5):
        self.count = 0

        # Opening video stream
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.step = int(self.fps/10)

        # Creating Queue
        self.Q = Queue(maxsize=queueSize)

        with open(model_dir + r'/config.json') as f:
            self.img_size = tuple(json.load(f)['image_size']['py/tuple'])

    def start(self):
        # Start thread to read frames
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            if self.stopped:
                return

            if not self.Q.full():
                # Read the next frame
                self.stream.set(1, self.count)
                grabbed, frame = self.stream.read()
                if not grabbed:
                    self.stopped = True
                    return
                frame = cv2.resize(frame, self.img_size)
                model_input = tf.expand_dims(frame, 0)
                self.Q.put((model_input, self.count))
                self.count += self.step

    def read(self):
        # Return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def run_detection_multi_thread(video_file, model_dir, project=None, save_dir=None, save=True, plot=False):
    # Setting basic variables
    if project is not None:

        delays = Delays()
        offset = - delays[project]
        # TODO: automatically find the project from the video
        # Get event information from excel
        print("Extracting excel data")
        excel = ex.extract_excel_data(r'data/' + project)
        vid_events = ex.extract_video_events(excel, video_file, offset)

    time_string = os.path.split(video)[-1].split('@')[0].split()[0]
    first_stamp = pd.to_datetime(time_string, format="%Y%m%d%H%M%S%f") - datetime.timedelta(seconds=offset)


    print("Starting video file thread...")
    fvs = FileVideoStream(video_file, model_dir).start()
    model = tf.keras.models.load_model(model_dir + '/saved_model')
    fps = fvs.fps

    # TODO: automation of creating and predicting mor classes here
    class_dict = {'FJOK': [], 'NONE': []}

    prob_dict = class_dict
    prob_dict['timestamp'] = []
    start = time.time()
    while fvs.more():
        # Read input tesnor for model
        model_input, frame_nr = fvs.read()
        timestamp = first_stamp + datetime.timedelta(seconds=(frame_nr/fps))
        # Run inference on tensor:
        pred = model(model_input).numpy()[0]

        if not type(model.layers[-1]) == tf.keras.layers.Softmax:
            pred = tf.nn.softmax(pred)
        # TODO: make sure class columns reflect possible other classes too
        prob_dict['FJOK'].append(pred[0])
        prob_dict['NONE'].append(pred[1])
        prob_dict['timestamp'].append(timestamp)

    prob = pd.DataFrame(prob_dict)
    end = time.time()
    print("execution time: {}".format(end-start))
    # calculating running mean
    # prob['FJOK_runningmean'] = prob.FJOK.rolling(window).mean()

    # saving findings to csv
    vid_name = os.path.split(video_file)[-1].split('.')[0]
    # TODO: add conditional statement if predictions have already been done
    if save and save_dir is not None:
        prob.to_csv(save_dir + '/prediction_' + vid_name + '.csv')
    elif save and save_dir is None:
        prob.to_csv(model_dir + '/prediction_' + vid_name + '.csv')

    if plot:
        # Plotting results
        plt.figure()
        plt.title("Probabilities")
        plt.plot(prob.timestamp, prob.FJOK, label='Prediction Field Joint')
        # plt.plot(range(len(prob['FJOK_runningmean'])), prob['FJOK_runningmean'], label='Rolling Average')
        # plt.vlines((vid_events['ms in video'] * (fps / 1000)).astype(int), 0, 1, linestyles='dashed', colors='r')
        plt.vlines(vid_events['datetime'], 0, 1, linestyles='dashed', colors='r', label="Event from listing")
        plt.xlabel("Frame")
        plt.ylabel("class probability")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help='Full path to the video file', required=True)
    parser.add_argument('--model', type=str, default=None, help="Full path to the directory "
                                                                "which contains the 'saved_model' directory",
                        required=True)
    parser.add_argument('--project', type=str, default=None, help='Name of the project the video belongs to',
                        required=True)
    parser.add_argument('--save', type=bool, default=True, help='Choice to store the predictions as a .csv file')
    parser.add_argument('--save_dir', type=str, default=None, help='Full path to location to store .csv file')
    parser.add_argument('--show', type=bool, default=False, help='choice to show plot of predictions')
    opt = parser.parse_args()

    if opt.source is None:
        video = os.path.abspath(r'data/video/20180115233422036@DVR-SD-01_Ch2_Trim3.mp4')
    elif os.path.isfile(opt.source):
        video = opt.source
    else:
        raise ValueError("Video not found in {}".format(opt.source))

    if opt.model is None:
        model_dir = os.path.abspath(r'runs/Varying layers and filters/127')
    elif os.path.isdir(opt.model + '/saved_model'):
        model_dir = opt.model
    else:
        raise ValueError("model not found in folder '{}' ".format(opt.model))

    projects = ["Troll", "Turkstream", "LingShui", "Nordstream", "Noble Tamar"]

    if opt.project is None:
        project = None
    elif opt.project not in projects:
        raise ValueError("unknown project specified, check spelling")
    else:
        project = opt.project

    if opt.save_dir is None:
        save_dir = None
    elif os.path.exists(opt.save_dir):
        save_dir = opt.save_dir
    else:
        os.makedirs(opt.save_dir)
        save_dir = opt.save_dir

    run_detection_multi_thread(video, model_dir, project, save_dir=save_dir, save=opt.save, window=60, plot=opt.show)
