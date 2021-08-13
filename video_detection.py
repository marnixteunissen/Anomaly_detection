from threading import Thread
from cv2 import VideoCapture, CAP_PROP_FPS, resize, CAP_PROP_FRAME_COUNT
from queue import Queue
from time import time, sleep
# from tensorflow import expand_dims, compat, queue, float32, int32, uint8
from tensorflow.keras.models import load_model
from json import load
from pandas import DataFrame, to_datetime
from math import floor
import os
import argparse
from datetime import timedelta
from TF_queue import TF_Queue
# from opencv_read_and_queue import OCV_stream


def run_detection_multi_thread(video_file, model_dir, save_dir=None, save=True):
    time_string = os.path.split(video)[-1].split('@')[0].split()[0]
    first_stamp = to_datetime(time_string, format="%Y%m%d%H%M%S%f")

    print("Loading Model...", flush=True)
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
    print(f'Loading time model: {stop-start}', flush=True)

    opencv_stream = OCV_stream(video_file, model_dir).start()
    fps = opencv_stream.fps

    tf_queue = TF_Queue(opencv_stream, model_dir).start()

    # TODO: automation of creating and predicting more classes here
    class_dict = {'FJOK': [], 'NONE': []}

    prob_dict = class_dict
    prob_dict['timestamp'] = []

    start = time()
    skipped_publications = 0
    publications_to_skip = 50

    #modelProcesingTimeStart = time()
    print("Starting inference...", flush=True)
    while tf_queue.more():
        # Read input tensor for model
        #modelProcesingTime += time()- modelProcesingTimeStart
        model_input, frame_nr = tf_queue.read()
        if frame_nr < 0:
            break
            
        frame_nr = int(frame_nr[0])
        # Run inference on tensor:
        pred = model(model_input).numpy()[0]
        timestamp = first_stamp + timedelta(seconds=(frame_nr / fps))

        # TODO: make sure class columns reflect possible other classes too
        prob_dict['FJOK'].append(pred[0])
        prob_dict['NONE'].append(pred[1])
        prob_dict['timestamp'].append(timestamp)

        if skipped_publications > (publications_to_skip - 1):
            print("Progress: {:.2f} %".format((100.0 * frame_nr) / opencv_stream.frame_count), flush=True)
            #print("Frames in queue: %d/%d" % (tf_queue.elements_in_ocv_queue(), tf_queue.elements_in_tf_queue()))        
            skipped_publications = 0
        else:
            skipped_publications += 1

    prob = DataFrame(prob_dict)
    end = time()
    print("execution time: {}".format(end-start))
    
    model_path_array = model_dir.split("\\")
    model_name = model_path_array[len(model_path_array)-2]
    model_number = model_path_array[len(model_path_array)-1]
    model_identf = model_name + '_' + model_number
    # saving findings to csv
    vid_name = os.path.split(video_file)[-1].split('.')[0]
    # TODO: add conditional statement if predictions have already been done
    if save and save_dir is not None:
        fullFileName = save_dir + '/' + model_identf + '_prediction_' + vid_name + '.csv'
    elif save and save_dir is None:
        fullFileName = model_dir + '/' + model_identf + '_prediction_'  + vid_name + '.csv'
    prob.to_csv(fullFileName)
    print("\n Saved to: " + fullFileName)


class OCV_stream:
    def __init__(self, path, model_dir, queueSize=16):
        # Counting total frames to keep track of progress
        self.frame_count = self.count_frames(path)

        # Opening video stream
        self.stream = VideoCapture(path)
        self.stopped = False
        self.fps = self.stream.get(CAP_PROP_FPS)
        self.frame_skip = max(floor(self.fps / 10 - 1), 0)
        self.step = 1 + self.frame_skip

        self.curr_frame = 0
        self.stream.set(1, 0)

        # Creating Queue
        self.Q = Queue(maxsize=queueSize)

        # get the image size from the
        with open(model_dir + r'/config.json') as f:
            self.img_size = tuple(load(f)['image_size']['py/tuple'])

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
                sleep(0.1)
                return

            if not self.Q.full():
                # Read the next frame
                grabbed = self.stream.grab()
                if not grabbed:
                    self.stop()
                    return
                if self.curr_frame % self.step == 0:
                    _, frame = self.stream.retrieve()
                    self.Q.put((frame, self.curr_frame))
                self.curr_frame += 1

            else:
                sleep(0.01)

    def read(self):
        # Return next frame in the queue
        return self.Q.get()

    def more(self):
        return not (self.Q.qsize() == 0 and self.stopped)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.Q.put((None, -1))

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


if __name__ == "__main__":
    print("Parsing arguments...", flush=True)
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
