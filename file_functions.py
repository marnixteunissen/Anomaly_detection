import os
import numpy


def set_proj_dir():
    dir_path = []


def get_video_dir():
    video_dir = []
    return video_dir


def get_video_file_names(video_dir, channels):
    dir_files = os.listdir(video_dir)
    video_files = [x for x in dir_files if x.endswith('.mp4')]
    files = []
    for channel in channels:
        for video_file in video_files:
            if video_file.endswith('Ch' + str(channel) + '.mp4'):
                files.append(video_file)
    if any(channels) not in [1, 2, 3, 4]:
        raise ValueError('Not all channels were found or they do not exist')

    return files


if __name__ == "__main__":
    file_list = get_video_file_names('C:\\Users\\MTN\\Documents\\Survey_anomaly_detection\\pycharm'
                         '\\Anomaly_detection\\data\\DATA_20200423153202169', [1])
    print(file_list)


