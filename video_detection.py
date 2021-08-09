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
from TF_queue import TF_Queue
from opencv_read_and_queue import OCV_stream


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
