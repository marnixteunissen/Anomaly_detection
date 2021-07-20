import os
import cv2
import excel_functions as excel_f
import file_functions as file
from tqdm import tqdm
import random


def extract_all_pos_frames(project, video_dir, excel_in, out_dir,
                           delay=0.000, channel=1, show=False, n_augment=0):
    """
    Extracts all the frames at all event times in given video for all given channels
    :param project:     string
                        name of the project
    :param video_dir:   path (string)
                        path to the directory containing the video
    :param excel_in:    dataframe
                        Dataframe extracted with extract_excel_data function
    :param out_dir:     path (string)
                        Directory to store snapshots
    :param delay:       float, Default: 0.000 sec
                        Time correction for video
    :param channel:     int, Default: 1 ('TOP')
                        the channel to extract frames from
    :param show:        bool, Default: False
                        bool to enable showing each picture as its being saved
                        if enabled will wait for a key for every frame.
    :return:
    """
    # Open excel data
    excel_data = excel_f.extract_video_events(excel_in, video_dir, static_offset=-delay)

    # Create iterable lists for creating frames
    time_stamps = excel_data["ms in video"].tolist()
    codes = excel_data["Secondary Code"].tolist()
    sample_nrs = excel_data.index.to_list()

    step = 100
    adding_stamps = []
    adding_codes = []
    adding_idx = []
    last_idx = sample_nrs[-1]

    for n in range(n_augment):
        tstep = step*((-1)**n)
        extra_stamps = [time+tstep for time in time_stamps]
        adding_stamps.extend(extra_stamps)
        adding_codes.extend(codes)
        extra_idx = [idx + (n+1)*last_idx for idx in sample_nrs]
        adding_idx.extend(extra_idx)
        if (n+1) % 2 == 0:
            step = 2 * step

    time_stamps.extend(adding_stamps)
    codes.extend(adding_codes)
    sample_nrs.extend(adding_idx)

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

    video_file = file.get_video_file_name(video_dir, channel)

    print("Opening", os.path.join(video_dir, video_file))
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    assert cap.isOpened()

    for (time_stamp, code, sample_nr) in tqdm(zip(time_stamps, codes, sample_nrs), total=len(time_stamps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp))
        success, image = cap.read()
        if success:
            save_path = os.path.join(save_dir, code, (str(sample_nr).zfill(6) + '_' + project + '.png'))
            cv2.imwrite(save_path, image)

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
    """
    Extracts all the frames at all event times in given video for all given channels
    :param project:     string
                        name of the project
    :param video_dir:   path (string)
                        path to the video directory
    :param excel_in:    dataframe
                        Dataframe extracted with extract_excel_data function
    :param out_dir:     path (string)
                        Directory to store snapshots
    :param delay:       float, Default: 0.000 sec
                        Time correction for video
    :param nr_samples:  int, Default: 5
                        number of negative samples between positive events
    :param interval:    int, Default: 4000 ms
                        ms interval around event not to sample as negative
    :param channel:     int, Default: 2
                        channel to extract the frames from
    :param show:        bool, Default: False
                        bool to enable showing each picture as its being saved
                        if enabled will wait for a key for every frame.
    """
    # Open excel data
    excel_data = excel_f.extract_video_events(excel_in, video_dir, static_offset=-delay)

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

    video_file = file.get_video_file_name(video_dir, channel)

    # Creating list of positive timestamps
    time_stamps = excel_data["ms in video"].tolist()

    # Create list of negative timestamps:
    neg_stamps = []
    for n in range(len(time_stamps)-1):
        # set range for negative samples:
        # min = stamp n + 4 sec, max = stamp (n+1) - 4 sec
        ms_min = min(time_stamps[n], time_stamps[n+1]) + interval
        ms_max = max(time_stamps[n], time_stamps[n+1]) - interval
        for i in range(nr_samples-1):
            if ms_min < ms_max:
                neg_stamps.append(random.randint(ms_min, ms_max))

    code = 'NONE'
    sample_nrs = [(x + excel_data.index[-1] + 1) for x in range(len(neg_stamps))]
    nr_success = 0

    # Open video file
    print("Opening", os.path.join(video_dir, video_file))
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    if not os.path.exists(save_dir):
        raise FileNotFoundError('Channel directories were not created')

    for (time_stamp, sample_nr) in tqdm(zip(neg_stamps, sample_nrs), total=len(neg_stamps)):
        cap.set(cv2.CAP_PROP_POS_MSEC, int(time_stamp))
        success, image = cap.read()
        if success:
            save_path = os.path.join(save_dir, code, (str(sample_nr).zfill(6) + '_' + project + '.png'))
            cv2.imwrite(save_path, image)
            nr_success = nr_success + 1
            if show:
                cv2.imshow("saved image:", image)
                cv2.waitKey()
        if not success:
            raise ValueError('Image was not read')

    print("Saved {}/{} images to {}".format(nr_success, len(neg_stamps), save_dir))
    print('')


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
    dir = r'K:\PROJECTS\SubSea Detection\12 - Data'
    print("working directory: ", dir)
    video = (dir + r"\Turkstream\Video_PL4_KP475-500\DATA_20180115220421906")
    excel = excel_f.extract_excel_data(dir + r'\Turkstream')
    chann = 2
    extract_all_pos_frames('Turkstream', video, excel, out_dir=r'C:\Users\MTN\PycharmProjects\Survey_anomaly_detection\pycharm\Anomaly_detection\data\test',
                           delay=0.000, channel=chann, augment=False, show=False, n_augment=4)
    # extract_all_neg_frames(dir + r'\data\Troll\video\DATA_20200423140158475', excel, dir + r'\data\Samples',
    #                        delay=0.500, channels=chann)

