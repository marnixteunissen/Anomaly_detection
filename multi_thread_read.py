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
        self.frame_skip = 10
        # Counting total frames to keep track of progress
        self.frame_count = self.count_frames(path)

        # Opening video stream
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.step = int(self.fps/self.frame_skip)

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

    def count_frames(self, path, override=False):
        # grab a pointer to the video file and initialize the total
        # number of frames read
        video = cv2.VideoCapture(path)
        total = 0

        # if the override flag is passed in, revert to the manual
        # method of counting frames
        if override:
            total = self.count_frames_manual(video)

        # otherwise, let's try the fast way first
        else:
            # lets try to determine the number of frames in a video
            # via video properties; this method can be very buggy
            # and might throw an error based on your OpenCV version
            # or may fail entirely based on your which video codecs
            # you have installed
            try:
                    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            # uh-oh, we got an error -- revert to counting manually
            except:
                total = self.count_frames_manual(video)

        # release the video file pointer
        video.release()

        # return the total number of frames in the video
        return total

    def count_frames_manual(video):
        # initialize the total number of frames read
        total = 0

        # loop over the frames of the video
        while True:
            # grab the current frame
            (grabbed, frame) = video.read()
         
            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break

            # increment the total number of frames read
            total += 1

        # return the total number of frames in the video file
        return total


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
    else:
        offset = 0

    time_string = os.path.split(video)[-1].split('@')[0].split()[0]
    first_stamp = pd.to_datetime(time_string, format="%Y%m%d%H%M%S%f") - datetime.timedelta(seconds=offset)

    print("Starting video file thread...")
    fvs = FileVideoStream(video_file, model_dir).start()
    if 'best' in os.listdir(model_dir + '/saved_model'):
        model = tf.keras.models.load_model(model_dir + '/saved_model/best')
    else:
        model = tf.keras.models.load_model(model_dir + '/saved_model')
    fps = fvs.fps

    # TODO: automation of creating and predicting mor classes here
    class_dict = {'FJOK': [], 'NONE': []}

    prob_dict = class_dict
    prob_dict['timestamp'] = []
    start = time.time()
    skipped_publications = 0
    publications_to_skip = 20
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
        
        if skipped_publications > (publications_to_skip - 1):
            print("Progress: {:.2f} %".format((100.0 * frame_nr) / fvs.frame_count))
            skipped_publications = 0
        else:
            skipped_publications += 1

    prob = pd.DataFrame(prob_dict)
    end = time.time()
    print("execution time: {}".format(end-start))

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
    parser.add_argument('--project', type=str, default=None, help='Name of the project the video belongs to')
    parser.add_argument('--save', type=str, default='True', help='Choice to store the predictions as a .csv file')
    parser.add_argument('--save_dir', type=str, default=None, help='Full path to location to store .csv file')
    parser.add_argument('--show', type=str, default='False', help='choice to show plot of predictions')
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

    show_output = (opt.show.lower() == 'true')
    save_output = (opt.save.lower() == 'true')

    run_detection_multi_thread(video, model_dir, project, save_dir=save_dir, save=save_output, plot=show_output)
