import os
import numpy
import random
import pandas as pd

# Channels are different per projects:
#CHANNELS = pd.DataFrame({'POS': ['LEFT', 'TOP', 'RIGHT'],
#                         'LingShui':    [3, 2, 4],
#                         'Troll':       [1, 2, 3],
#                         'Turkstream':  [3, 2, 4]})


def ch(project, channels):
    """
    Maps the given video channels to the videostreams in the project.
    :param project: str: name of the project
    :param channels: int, list of int. int referring to the channel. left=0, top=1, right=2
    :return:    project_channels, int, list of int, corresponding to channels in project.
                channels_strings, str, list of str, from ['LEFT', 'TOP', 'RIGHT']
    """
    # Each project has the channels distributed in a different way:
    proj_ch = pd.DataFrame({'POS': ['LEFT', 'TOP', 'RIGHT'],
                             'LingShui':    [3, 2, 4],
                             'Troll':       [1, 2, 3],
                             'Turkstream':  [3, 2, 4]})
    if not type(channels) == list:
        channels = [channels]
    # create a list with channel numbers
    proj_channels = proj_ch[project].loc[channels].to_list()
    # create a list of position strings
    ch_str = proj_ch['POS'].loc[channels].to_list()

    return proj_channels, ch_str


def get_video_file_names(project, video_dir, channel_idx):

    # Get the correct channel name from the project
    channels, ch_str = ch(project, channel_idx)

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


def get_video_file_name(video_dir, channel):
    dir_files = os.listdir(video_dir)
    video_file = [x for x in dir_files if x.endswith(str(channel) + '.mp4')]

    return video_file[0]


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


def create_channel_folders(root_dir, classes=["FJOK", "NONE"]):
    """
    Creates folder structure for saving the data-set
    :param root_dir: path to folder where the dataset will be saved
    :param classes: list of str, containing the classes. Default: ["FJOK", "NONE"]
    :return:
    """
    for channel in ['LEFT', 'TOP', 'RIGHT']:
        if not os.path.exists(os.path.join(root_dir, channel)):
            os.mkdir(os.path.join(root_dir, channel))
        for mode in ['train', 'test']:
            if not os.path.exists(os.path.join(root_dir, channel, mode)):
                os.mkdir(os.path.join(root_dir, channel, mode))
            for c in classes:
                if not os.path.exists(os.path.join(root_dir, channel, mode, c)):
                    os.mkdir(os.path.join(root_dir, channel, mode, c))


if __name__ == "__main__":
    # root_dir = r'K:\PROJECTS\SubSea Detection\12 - Data'
    # project_dir = os.path.join(root_dir, 'LingShui')
    # train_data_dirs, test_data_dirs = train_test_split(project_dir)
    # print(train_data_dirs)
    # print(test_data_dirs)
    channel, string = ch('Turkstream', [2, 0])
    print('string =', string)
    print('channel nr =', channel)


