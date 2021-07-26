from threading import Thread
from cv2 import VideoCapture, CAP_PROP_FPS, resize, CAP_PROP_FRAME_COUNT
from queue import Queue
from time import time, sleep
from tensorflow import expand_dims, compat, queue, float32, int32, uint8
from tensorflow.keras.models import load_model
from json import load
from pandas import DataFrame, to_datetime
from math import floor
import os
import argparse
from datetime import timedelta


class TF_Queue:
    def __init__(self, ocv_queue, model_dir, queuesize=16):
        self.stopped = False
        self.ocv_Q = ocv_queue
        # Creating Queue
        with open(model_dir + r'/config.json') as f:
            # this size is in opencv format: tf standard tensor format: [H, W]
            self.img_size = tuple(load(f)['image_size']['py/tuple'])
        # here shape is according to tf standard tensor format: [n, H, W, C]
        self.tf_queue = queue.FIFOQueue(queuesize, [uint8, int32], shapes=[[1, self.img_size[0], self.img_size[1], 3], [1]])

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

            # Read the next frame
            frame, count = self.ocv_Q.read()

            if not self.ocv_Q.more():
                self.stopped = True
                return

            # resizing the frame using the opencv order: [W, H]
            frame = resize(frame, [self.img_size[1], self.img_size[0]])
            model_input = expand_dims(frame, 0)

            self.tf_queue.enqueue([model_input, [count]])

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
