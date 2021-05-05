import os
import numpy
import random


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


def train_test_split(project_dir, test_split=0.2, startstr='Video', part=1.0):
    for entry in os.listdir(project_dir):
        if entry.startswith(startstr) or entry.startswith('video'):
            video_dir = os.path.join(project_dir, entry)
    if video_dir is not None:
        data_dirs = os.listdir(video_dir)
        data_dirs = [x for x in data_dirs if 'DATA' in x]
    else:
        raise FileNotFoundError("No folder starting with 'Video', 'video' or specified string was found")

    random.shuffle(data_dirs)
    nr_used = int(part * len(data_dirs))
    nr_test = int(test_split * nr_used)

    test_data_dirs = data_dirs[:nr_test]
    train_data_dirs = data_dirs[nr_test:]

    return train_data_dirs, test_data_dirs


if __name__ == "__main__":
    file_list = get_video_file_names('C:\\Users\\MTN\\Documents\\Survey_anomaly_detection\\pycharm'
                         '\\Anomaly_detection\\data\\DATA_20200423153202169', [1])
    print(file_list)


