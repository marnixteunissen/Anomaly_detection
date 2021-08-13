from threading import Thread
from cv2 import VideoCapture, CAP_PROP_FPS, resize, CAP_PROP_FRAME_COUNT
from queue import Queue
from time import time, sleep
# import tensorflow as tf
import json
# import pandas as pd
from math import floor
# import os
# import excel_functions as ex
# import matplotlib.pyplot as plt
# import datetime
# import argparse


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
