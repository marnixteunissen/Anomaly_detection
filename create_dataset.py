from file_functions import *
from excel_functions import *
from opencv_functions import *
import os
from random import shuffle
from shutil import move
import argparse


def create_dataset(project,
                   channel_idx=1,
                   classes='Field Joint',
                   delay=0.000,
                   neg_samples=5,
                   extra_pos_samples=0,
                   out_dir='',
                   split=0.15,
                   use_perc=1.0,
                   root_dir=r'K:\PROJECTS\SubSea Detection\12 - Data'):

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
                extract_all_event_frames(project, video_dir, data_points, output_dir,
                                         delay=delay, channel=channel, n_augment=extra_pos_samples)
                tot_neg = (extra_pos_samples + 1) * neg_samples
                extract_all_neg_frames(project, video_dir, data_points, output_dir,
                                       nr_samples=tot_neg, delay=delay, channel=channel)

    print('Data-set created for project', project)


def create_new_dataset(project,
                       channel_idx=1,
                       classes=['FJOK'],
                       delay=0.000,
                       neg_samples=5,
                       extra_pos_samples=0,
                       out_dir='',
                       test_split=0.15,
                       root_dir=r'K:\PROJECTS\SubSea Detection\12 - Data'):

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

    # Setting output directory:
    yes = {'yes', 'y', 'ye', ''}

    if out_dir == '' and os.path.exists(os.path.join(root_dir, 'data-set')):
        save_dir = os.path.join(root_dir, 'data-set')
        create_channel_folders(save_dir, classes)

    elif out_dir == '' and not os.path.exists(os.path.join(root_dir, 'data-set')):
        choice = input("Would you like to create a 'data-set' folder in '{}'?".format(root_dir)).lower()
        if choice in yes:
            save_dir = os.path.join(root_dir, 'data-set')
            os.mkdir(save_dir)
            create_channel_folders(save_dir, classes)
        else:
            raise Exception('Data-set directory was not created, no data was extracted.')

    elif os.path.exists(out_dir):
        save_dir = out_dir
        create_channel_folders(save_dir, classes)

    else:
        choice = input('Specified output directory not found, would you like to create it? (y)/(n)')
        if choice in yes:
            os.mkdir(out_dir)
            save_dir = out_dir
            create_channel_folders(save_dir)

        else:
            raise Exception('Specified save directory not found, and not created')

    print('Saving dataset to:', save_dir)

    data_points = get_event_types(project_dir, event_types=classes)
    data_dirs = get_video_dirs(project_dir)

    # Create the full data-set in train folder:
    print('channels:', str(ch_str))
    for channel_string, channel in zip(ch_str, channels):
        channel_dir = os.path.join(save_dir, channel_string)
        for dir in data_dirs:
            output_dir = os.path.join(channel_dir, 'train')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            extract_all_event_frames(project, dir, data_points, output_dir,
                                     delay=delay, channel=channel, n_augment=extra_pos_samples)
            tot_neg = (extra_pos_samples + 1) * neg_samples
            extract_all_neg_frames(project, dir, data_points, output_dir,
                                   nr_samples=tot_neg, delay=delay, channel=channel)

        # move part of data to test folder:
        print('Moving test files for {} channel'.format(channel_string))
        for cl in classes:
            samples = [sample for sample in os.listdir(os.path.join(channel_dir, 'train', cl)) if sample.endswith(project + ".png")]
            n_samples = len(samples)
            print(n_samples)
            n_test = int(test_split * n_samples)
            print(n_test)
            shuffle(samples)
            for file in samples[:n_test]:
                source = os.path.join(channel_dir, 'train', cl, file)
                destination = os.path.join(channel_dir, 'test', cl, file)
                # move file
                move(source, destination)


if __name__ == "__main__":
    # TODO: add parser arguments for commandline running
    delays = {'LingShui':           1.900,
              'Troll':              2.250,
              'Turkstream':         0.500,
              'Baltic Connector':   0.500,
              'Noble Tamar':        2.850,
              'Nordstream':         1.200,
              'Sur de Texas':       0.000,
              'Tulip Oil':          1.600}

    dataset_dir = r'E:\dataset_22_07_21'
    root = 'Other'
    if root == 'H':
        # for projects on H: drive
        projects = ['Baltic Connector', 'Noble Tamar', 'Nordstream']
        extras = [7, 22, 0]
        negatives = [6, 6, 6]
        data_dir = r'H:\Data'
    elif root == 'K':
        # for projects on K: drive
        projects = ['Troll', 'Turkstream']
        extras = [11, 7]
        negatives = [6, 6]
        data_dir = r'K:\PROJECTS\SubSea Detection\12 - Data'
    elif root == 'Other':
        projects = ['Tulip Oil']
        extras = [0]
        negatives = [0]
        data_dir = r'C:\Users\MTN\Documents\Anomaly_Detection\Data'
    else:
        raise ValueError('Wrong root')

    create_new_dataset(projects[0],
                       channel_idx=1,
                       classes=['FJOK'],
                       delay=delays[projects[0]],
                       neg_samples=negatives[0],
                       extra_pos_samples=extras[0],
                       out_dir='',
                       test_split=0.15,
                       root_dir=data_dir)
    # for project, extra, negative in zip(projects, extras, negatives):
    #     delay = delays[project]
    #     create_dataset(project,
    #                    delay=delay,
    #                    neg_samples=negative,
    #                    extra_pos_samples=extra,
    #                    root_dir=data_dir,
    #                    out_dir=dataset_dir)
