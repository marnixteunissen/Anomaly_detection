import os
import cv2
import excel_functions as excel_f
import file_functions as file
from tqdm import tqdm
import random
import numpy as np


def extract_all_event_frames(project, video_dir, excel_in, out_dir,
                             delay=0.000, channel=1, show=False, n_augment=0, test=False):
    # Open excel data
    excel_data = excel_f.extract_video_events(excel_in, video_dir, static_offset=-delay)
    video_file = file.get_video_file_name(video_dir, channel)
    print("Opening", os.path.join(video_dir, video_file))

    # Create iterable lists for creating frames
    time_stamps = excel_data["ms in video"].tolist()
    codes = excel_data["code"].tolist()
    sample_nrs = excel_data.index.to_list()

    if n_augment <= 10:
        step = 100
    elif 10 < n_augment < 25:
        step = 75
    else:
        raise(ValueError('Too many extra samples selected, please stay within the [0 : 25] range'))

    adding_stamps = []
    adding_codes = []
    adding_idx = []
    if len(sample_nrs) == 0:
        last_idx = 0
    else:
        last_idx = sample_nrs[-1]

    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    assert cap.isOpened(), 'Video was not opened'
    total_ms = cap.get(cv2.CAP_PROP_FRAME_COUNT) * 1000 / cap.get(cv2.CAP_PROP_FPS)

    for n in range(n_augment):
        tstep = step*((-1)**n)
        np_stamps = np.array(time_stamps)
        extra_stamps = [time+tstep for time in time_stamps]

        valid_idx = [idx for idx, stamp in enumerate(extra_stamps) if (total_ms > stamp >= 0)]
        adding_stamps.extend([extra_stamps[idx] for idx in valid_idx])
        adding_codes.extend([codes[idx] for idx in valid_idx])

        extra_idx = [idx + (n+1)*last_idx for idx in range(len(valid_idx))]
        # extra_idx = [idx + (n+1)*last_idx for idx in sample_nrs]
        adding_idx.extend(extra_idx)
        if (n+1) % 2 == 0:
            step = 2 * step

    time_stamps.extend(adding_stamps)
    codes.extend(adding_codes)
    sample_nrs.extend(adding_idx)
    print('Creating {} samples'.format(len(sample_nrs)))
    nr_success = 0

    # Set output directory:
    working_dir = os.getcwd()
    if out_dir == '':
        save_dir = working_dir
    elif os.path.exists(out_dir):
        save_dir = out_dir
    else:
        yes = {'yes', 'y', 'ye', ''}
        choice = input(('Path does not exist, would you like to create' + out_dir + '? (y)/(n)')).lower()
        if choice in yes:
            os.mkdir(out_dir)
            save_dir = out_dir
        else:
            raise FileNotFoundError('Directory ', out_dir, ' was not found and not created, '
                                                           'please use existing directory or create one')

    for (time_stamp, code, sample_nr) in tqdm(zip(time_stamps, codes, sample_nrs), total=len(time_stamps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp))
        success, image = cap.read()
        nr_files_before = len(os.listdir(os.path.join(save_dir, code)))
        if success:
            save_path = os.path.join(save_dir, code, (str(sample_nr).zfill(6) + '_' + project + '.png'))
            cv2.imwrite(save_path, image)

            nr_files_after = len(os.listdir(os.path.join(save_dir, code)))
            if nr_files_after-nr_files_before != 0:
                nr_success = nr_success + 1

            if show:
                cv2.imshow("saved image:", image)
                cv2.waitKey()
        if not success:
            raise ValueError('Image was not read')

    print("Saved {}/{} images to {}".format(nr_success, len(time_stamps), save_dir))
    print('')


def extract_all_neg_frames(project, video_dir, excel_in, out_dir,
                           delay=0.000, nr_samples=5, interval=3000, channel=2, show=False):
    # Open excel data
    excel_data = excel_f.extract_video_events(excel_in, video_dir, static_offset=-delay)

    video_file = file.get_video_file_name(video_dir, channel)
    print("Opening", os.path.join(video_dir, video_file))

    # Set output directory
    working_dir = os.getcwd()
    if out_dir == '':
        save_dir = working_dir
    elif os.path.exists(out_dir):
        save_dir = out_dir
    else:
        yes = {'yes', 'y', 'ye', ''}
        choice = input(('Path does not exist, would you like to create' + out_dir + '? (y)/(n)')).lower()
        if choice in yes:
            os.mkdir(out_dir)
            save_dir = out_dir
        else:
            raise FileNotFoundError('Directory ', out_dir, ' was not found and not created, '
                                                           'please use existing directory or create one')

    # Creating list of positive timestamps
    time_stamps = excel_data["ms in video"].tolist()

    # Create list of negative timestamps:
    neg_stamps = []
    for n in range(len(time_stamps)-1):
        # set range for negative samples:
        ms_min = min(time_stamps[n], time_stamps[n+1]) + interval
        ms_max = max(time_stamps[n], time_stamps[n+1]) - interval
        for i in range(nr_samples-1):
            if ms_min < ms_max:
                neg_stamps.append(random.randint(ms_min, ms_max))

    code = 'NONE'
    target_dir = os.path.join(save_dir, code)
    files_in_target = os.listdir(target_dir)

    if len(files_in_target) == 0:
        highest_sample_nr = 0
    else:
        project_files = [int(x.split('_')[0]) for x in files_in_target if x.split('_')[-1] == (project + '.png')]
        if len(project_files) == 0:
            highest_sample_nr = 0
        else:
            highest_sample_nr = max(project_files)

    sample_nrs = [(x + highest_sample_nr + 1) for x in range(len(neg_stamps))]

    nr_files_before = len(files_in_target)

    # Open video file
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    if not os.path.exists(save_dir):
        raise FileNotFoundError('Channel directories were not created')

    for (time_stamp, sample_nr) in tqdm(zip(neg_stamps, sample_nrs), total=len(neg_stamps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp))
        success, image = cap.read()
        if success:
            save_path = os.path.join(save_dir, code, (str(sample_nr).zfill(6) + '_' + project + '.png'))
            cv2.imwrite(save_path, image)
            if show:
                cv2.imshow("saved image:", image)
                cv2.waitKey()
        if not success:
            raise ValueError('Image was not read')

    nr_files_after = len(os.listdir(target_dir))

    print("Saved {}/{} images to {}".format((nr_files_after-nr_files_before), len(neg_stamps), save_dir + r'\NONE'))
    print('')


# unused extra functions:
def get_first_frame(video_file):
    """Returns the first frame of a video as an image"""
    cap = cv2.VideoCapture(video_file)
    success, image = cap.read()
    if not success:
        raise ValueError('Image was not read')
    return image


def get_last_frame(video_file):
    """Returns the last frame of a video as an image"""
    vid_cap = cv2.VideoCapture(video_file)
    frame_nr = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr-1)
    success, image = vid_cap.read()
    if not success:
        raise ValueError('Image was not read')
    return image, frame_nr


def extract_frame(video_file, time=500):
    """Returns the frame at the specified time as an image"""
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, time)
    label = []
    success, image = cap.read()
    if not success:
        raise ValueError('Image was not read')
    return image


if __name__ == "__main__":
    dir = r''
    video = r'C:\Users\MTN\Documents\Anomaly_Detection\Data\Tulip Oil\Video\DATA_20180620122949270'
    proj_dir = r'C:\Users\MTN\Documents\Anomaly_Detection\Data\Tulip Oil'
    excel =  excel_f.get_event_types(proj_dir, ['FJOK', "PBOK"])
    out = r'C:\Users\MTN\Documents\Anomaly_Detection\Anomaly_detection\data\test'
    extract_all_event_frames('Tulip Oil', video, excel, out, test=True, n_augment=0, channel=3, delay=1.2)
    extract_all_neg_frames('Tulip Oil', video, excel, out, channel=3, delay=1.2)

    # dir = r'K:\PROJECTS\SubSea Detection\12 - Data\Troll'
    # print("working directory: ", dir)
    # video = (dir + r"\Video Line 3\DATA_20200424010632051")
    # excel = excel_f.extract_excel_data(dir)
    # out_dir = r'C:\Users\MTN\PycharmProjects\Survey_anomaly_detection\pycharm\Anomaly_detection\data\test'
    # chann = 2
    # extract_all_pos_frames('Troll', video, excel, out_dir=out_dir,
    #                        delay=0.000, channel=chann, show=False)
    # extract_all_neg_frames('Troll', video, excel, out_dir=out_dir,
    #                        delay=0.000, nr_samples=5, interval=3000, channel=chann, show=False)

