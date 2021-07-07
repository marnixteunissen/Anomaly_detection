from threading import Thread
from cv2 import VideoCapture, CAP_PROP_FPS, resize, CAP_PROP_FRAME_COUNT
from queue import Queue
from time import time, sleep
from tensorflow import expand_dims, compat, queue, float32, int16, uint8
from tensorflow.keras.models import load_model
from json import load
from pandas import DataFrame, to_datetime
import os
import argparse
from datetime import timedelta


class FileVideoStream:
    def __init__(self, path, model_dir, queueSize=20, batch_size=32):
        self.count = 0
        self.frame_skip = 0
        self.batch_size = batch_size
        # Counting total frames to keep track of progress
        self.frame_count = self.count_frames(path)

        # Opening video stream
        self.stream = VideoCapture(path)
        self.stopped = False
        self.fps = self.stream.get(CAP_PROP_FPS)
        print(f"Framerate in video is {self.fps} fps, skipping {self.frame_skip} frames per iteration")
        self.step = 1 + self.frame_skip

        # Creating Queue
        self.Q = Queue(maxsize=queueSize)

        with open(model_dir + r'/config.json') as f:
            # this size is in opencv format: [W, H]
            self.img_size = tuple(load(f)['image_size']['py/tuple'])
        # here shape is according to tf standard tensor format: [n, H, W, C]
        self.tf_queue = queue.FIFOQueue(16, [uint8, int16], shapes=[[1, self.img_size[1], self.img_size[0], 3], [1]])

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

            self.stream.set(1, self.count)

            # Read the next frame
            grabbed, frame = self.stream.read()

            if not grabbed:
                self.stopped = True
                return

            frame = resize(frame, self.img_size)
            model_input = expand_dims(frame, 0)

            self.tf_queue.enqueue([model_input, [self.count]])
            self.count += self.step

    def read(self):
        # Return next frame in the queue
        return self.tf_queue.dequeue()

    def more(self):
        # return True if there are still frames in the queue
        tries = 0
        while self.tf_queue.size() == 0 and not self.stopped and tries < 5:
            sleep(0.1)
            tries += 1
        return self.tf_queue.size() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def count_frames(self, path, override=False):
        # grab a pointer to the video file and initialize the total
        # number of frames read
        video = VideoCapture(path)
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
                    total = int(video.get(CAP_PROP_FRAME_COUNT))

            # uh-oh, we got an error -- revert to counting manually
            except:
                total = self.count_frames_manual(video)

        # release the video file pointer
        video.release()

        # return the total number of frames in the video
        return total

    def count_frames_manual(self, video):
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


def run_detection_multi_thread(video_file, model_dir, save_dir=None, save=True):
    time_string = os.path.split(video)[-1].split('@')[0].split()[0]
    first_stamp = to_datetime(time_string, format="%Y%m%d%H%M%S%f")

    print("Loading Model...")
    start = time()
    if 'best' in os.listdir(model_dir + '/saved_model'):
        if 'best_model.h5' in os.listdir(model_dir + '/saved_model/best'):
            model = load_model(model_dir + '/saved_model/best/best_model.h5')
        else:
            model = load_model(model_dir + '/saved_model/best')
    else:
        if 'saved_model.h5' in os.listdir(model_dir + '/saved_model'):
            model = load_model(model_dir + '/saved_model.h5')
        else:
            model = load_model(model_dir + '/saved_model')
    stop = time()
    print(f'Loading time model: {stop-start}')

    print("Starting video file thread...")
    fvs = FileVideoStream(video_file, model_dir).start()
    fps = fvs.fps/fvs.step

    # TODO: automation of creating and predicting more classes here
    class_dict = {'FJOK': [], 'NONE': []}

    prob_dict = class_dict
    prob_dict['timestamp'] = []

    start = time()
    skipped_publications = 0
    publications_to_skip = 20

    print("Starting inference...")
    while fvs.more():
        # Read input tensor for model
        model_input, frame_nr = fvs.read()
        frame_nr = int(frame_nr[0])
        # Run inference on tensor:
        pred = model(model_input).numpy()[0]
        timestamp = first_stamp + timedelta(seconds=(frame_nr / fps))

        # TODO: make sure class columns reflect possible other classes too
        prob_dict['FJOK'].append(pred[0])
        prob_dict['NONE'].append(pred[1])
        prob_dict['timestamp'].append(timestamp)

        if skipped_publications > (publications_to_skip - 1):
            print("Progress: {:.2f} %".format((100.0 * frame_nr) / fvs.frame_count))
            skipped_publications = 0
        else:
            skipped_publications += 1

    prob = DataFrame(prob_dict)
    end = time()
    print("execution time: {}".format(end-start))

    # saving findings to csv
    vid_name = os.path.split(video_file)[-1].split('.')[0]
    # TODO: add conditional statement if predictions have already been done
    if save and save_dir is not None:
        prob.to_csv(save_dir + '/prediction_' + vid_name + '.csv')
    elif save and save_dir is None:
        prob.to_csv(model_dir + '/prediction_' + vid_name + '.csv')


if __name__ == "__main__":
    print("parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None,
                        help='Full path to the video file',
                        required=True)
    parser.add_argument('--model', type=str, default=None,
                        help="Full path to the directory which contains the 'saved_model' directory",
                        required=True)
    parser.add_argument('--save', type=str, default='True',
                        help='Choice to store the predictions as a .csv file')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Full path to location to store .csv file')
    opt = parser.parse_args()

    if os.path.isfile(opt.source):
        video = opt.source
    else:
        raise ValueError("Video not found in {}".format(opt.source))

    if os.path.isdir(opt.model + '/saved_model'):
        model_dir = opt.model
    else:
        raise ValueError("model not found in folder '{}' ".format(opt.model))

    if opt.save_dir is None:
        save_dir = None
    elif os.path.exists(opt.save_dir):
        save_dir = opt.save_dir
    else:
        os.makedirs(opt.save_dir)
        save_dir = opt.save_dir

    save_output = (opt.save.lower() == 'true')

    run_detection_multi_thread(video, model_dir, save_dir=save_dir, save=save_output)
