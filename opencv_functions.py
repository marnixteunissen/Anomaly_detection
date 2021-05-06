import Anomaly_detection
import os
import subprocess
import cv2
import Anomaly_detection.excel_functions as excel_f
import Anomaly_detection.file_functions as file


def extract_all_frames(video_dir, excel_in, out_dir, delay=0.000, channels=2, show=False):
    """
    Extracts all the frames at all event times in given video for all given channels
    :param video_dir:   path (string)
                        path to the video directory
    :param excel_in:    dataframe
                        Dataframe extracted with extract_excel_data function
    :param out_dir:     path (string)
                        Directory to store snapshots
    :param delay:       float
                        Time correction for video
    :param channels:    int, list of int
                        List of all the channels to extract frames from
    :return:
    """
    # Open excel data
    excel_data = excel_f.extract_video_events(excel_in, video_dir, static_offset=delay)

    # Create iterable lists for creating frames
    time_stamps = excel_data["ms in video"].tolist()
    codes = excel_data["Secondary Code"].tolist()
    sample_nrs = excel_data.index.to_list()

    # Set output directory:
    working_dir = os.getcwd()
    if out_dir == '':
        save_dir = working_dir
    elif os.path.exists(out_dir):
        save_dir = out_dir
    else:
        raise FileNotFoundError('Directory ', out_dir, ' was not found, please use existing directory')
    print("saving to:", save_dir)

    # Create list if channels is not a list
    if not type(channels) == list:
        channels = [channels]
    video_files = file.get_video_file_names(video_dir, channels)
    
    # open video files one by one:
    for video_file, channel in zip(video_files, channels):
        print("Opening", os.path.join(video_dir, video_file))
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        channel_dir = save_dir + r'/Channel_' + str(channel)
        
        for (time_stamp, code, sample_nr) in zip(time_stamps, codes, sample_nrs):
            cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp))
            success, image = cap.read()

            if success:
                save_path = os.path.join(channel_dir, (str(sample_nr) + '_' + code + '.png'))
                cv2.imwrite(save_path, image)
                print('Image was saved to', save_path)
                if show:
                    cv2.imshow("saved image:", image)
                    cv2.waitKey()
            if not success:
                raise ValueError('Image was not read')


def get_first_frame(video_file):
    cap = cv2.VideoCapture(video_file)
    success, image = cap.read()
    if not success:
        raise ValueError('Image was not read')
    return image


def get_last_frame(video_file):
    vid_cap = cv2.VideoCapture(video_file)
    frame_nr = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr-1)
    success, image = vid_cap.read()
    if not success:
        raise ValueError('Image was not read')
    return image, frame_nr


def extract_frame(video_file, time=500):
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, time)
    label = []
    success, image = cap.read()
    assert success
    if not success:
        raise ValueError('Image was not read')
    return image


if __name__ == "__main__":
    dir = os.getcwd()
    print("working directory: ", dir)
    video = (dir + r"\data\video\20200423213211791@MainDVR_Ch2.mp4")
    print("path to video file:", video)
    image3 = extract_frame(video, 500)
    #cv2.imshow('mid frame', image3)
    #cv2.waitKey(0)
    excel = excel_f.extract_excel_data(dir + r'\data\Troll', classes='Field Joint')

    #data = excel_f.extract_video_events(excel, dir + r'\data\Troll\video\DATA_20200423153202169')
    #print(data)
    extract_all_frames(dir + r'\data\Troll\video\DATA_20200423203210192', excel,
                       dir + r'\data\samples', delay=-2.000, channels=[1, 2, 3])

