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
                files.append(os.path.join(video_dir, video_file))
    if any(channels) not in [1, 2, 3, 4]:
        raise ValueError('Not all channels were found or they do not exist')

    return files


def train_test_split(project_dir, test_split=0.2, startstr='Video', part=1.0):
    """
    Split the data into training and test data
    :param project_dir: Path (string)
                        Path to the project directory
    :param test_split:  Float, Deflaut: 0.2
                        Percentage of the data used for testing
    :param startstr:    String, Default: 'Video'
                        Start string of the folder name containing the videos
    :param part:        Float, Default; 1.0
                        Percentage of the video data used to create the samples (train+test)
    :return:            Lists
                        Two listst of video directory names, training data, testing data
    """
    for entry in os.listdir(project_dir):
        if entry.startswith(startstr) or entry.startswith('video') or entry.startswith('Video'):
            video_dir = os.path.join(project_dir, entry)
    if video_dir is not None:
        data_dirs = os.listdir(video_dir)
        data_dirs = [os.path.join(video_dir, x) for x in data_dirs if 'DATA' in x]
    else:
        raise FileNotFoundError("No folder starting with 'Video', 'video' or specified string was found")

    random.shuffle(data_dirs)
    nr_used = int(part * len(data_dirs))
    nr_test = int(test_split * nr_used)

    test_data_dirs = data_dirs[:nr_test]
    train_data_dirs = data_dirs[nr_test:]

    return train_data_dirs, test_data_dirs


if __name__ == "__main__":
    root_dir = r'K:\PROJECTS\SubSea Detection\12 - Data'
    project_dir = os.path.join(root_dir, 'LingShui')
    train_data_dirs, test_data_dirs = train_test_split(project_dir)
    print(train_data_dirs)
    print(test_data_dirs)


