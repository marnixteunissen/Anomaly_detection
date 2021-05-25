from Anomaly_detection.file_functions import *
from Anomaly_detection.excel_functions import *
from Anomaly_detection.opencv_functions import *
import os
import argparse


def create_dataset(project,
                   channel_idx=1,
                   classes='Field Joint',
                   delay=0.000,
                   neg_samples=5,
                   out_dir='',
                   split=0.2,
                   use_perc=1.0,
                   root_dir=r'K:\PROJECTS\SubSea Detection\12 - Data'):

    # Check if project is available:
    if not project in ['LingShui', 'Troll', 'Turkstream']:
        raise FileNotFoundError('Project files were not found in %s'.format(root_dir))

    # Getting correct channel names:
    channels, ch_str = ch(project, channel_idx)

    # Check if root directory exists:
    if not os.path.exists(root_dir):
        raise FileExistsError('specified root directory does not exist')

    # Setting project directory
    project_dir = os.path.join(root_dir, project)
    if os.path.exists(project_dir):
        print('Extracting data samples from project:', project)
    else:
        raise FileNotFoundError('Specified project not found in', root_dir)

    # making the test/train split:
    train_data_dirs, test_data_dirs = train_test_split(project_dir, test_split=split, part=use_perc)

    # Setting output directory:
    yes = {'yes', 'y', 'ye', ''}

    if out_dir == '' and os.path.exists(os.path.join(root_dir, 'data-set')):
        save_dir = os.path.join(root_dir, 'data-set')
        create_channel_folders(save_dir)

    elif out_dir == '' and not os.path.exists(os.path.join(root_dir, 'data-set')):
        choice = input("Would you like to create a 'data-set' folder in '{}'?".format(root_dir)).lower()
        if choice in yes:
            save_dir = os.path.join(root_dir, 'data-set')
            os.mkdir(save_dir)
            create_channel_folders(save_dir)
        else:
            raise Exception('Data-set directory was not created, no data was extracted.')

    elif os.path.exists(out_dir):
        save_dir = out_dir
        create_channel_folders(save_dir)

    else:
        choice = input('Specified output directory not found, would you like to create it? (y)/(n)')
        if choice in yes:
            os.mkdir(out_dir)
            save_dir = out_dir
            create_channel_folders(save_dir)

        else:
            raise Exception('Specified save directory not found, and not created')

    print('Saving dataset to:', save_dir)

    # Extracting data-points from excel file:
    data_points = extract_excel_data(project_dir, classes=classes)

    # Create training and test data-sets:
    print('channels:', str(ch_str))
    for channel_string, channel in zip(ch_str, channels):
        channel_dir = os.path.join(save_dir, channel_string)
        for mode, dirs in zip(['train', 'test'], [train_data_dirs, test_data_dirs]):
            output_dir = os.path.join(channel_dir, mode)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            for video_dir in dirs:
                # extracting all the frames from the video (only Field Joints for now)
                extract_all_pos_frames(project, video_dir, data_points, output_dir,
                                       delay=delay, channel=channel)
                extract_all_neg_frames(project, video_dir, data_points, output_dir,
                                       nr_samples=neg_samples,
                                       delay=delay, channel=channel)

    print('Data-set created for project', project)


if __name__ == "__main__":
    # TODO: add parser arguments for commandline running
    # Delay LingShui:   1.900
    # Delay Troll:      1.550
    # Delay Turkstream: ???
    root = os.getcwd() + r'\data'
    create_dataset('Troll', delay=1.550, neg_samples=6,
                   root_dir=root, use_perc=0.1)
    create_dataset('Troll', delay=1.900, neg_samples=6,
                   root_dir=root)