from Anomaly_detection.file_functions import *
from Anomaly_detection.excel_functions import *
from Anomaly_detection.opencv_functions import *
import os


# steps:
# Set root directory:
# Select project:
# Make train-test split:
# Set location to store test and training data

def create_dataset(project,
                   channels = [2],
                   classes = 'Field Joint',
                   delay = 0.000,
                   out_dir = '',
                   split = 0.2,
                   use_perc = 1.0,
                   root_dir = r'K:\PROJECTS\SubSea Detection\12 - Data'):
    """

    :param project:
    :param channels:
    :param classes:
    :param delay:
    :param out_dir:
    :param split:
    :param use_perc:
    :param root_dir:
    :return:
    """
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
    no = {'no', 'n'}

    if out_dir == '' and os.path.exists(os.path.join(root_dir, 'data-set')):
        save_dir = os.path.join(root_dir, 'data-set')
        if not os.path.exists(os.path.join(save_dir, 'train')):
            os.mkdir(os.path.join(save_dir, 'train'))
        if not os.path.exists(os.path.join(save_dir, 'test')):
            os.mkdir(os.path.join(save_dir, 'test'))

    elif out_dir == '' and not os.path.exists(os.path.join(root_dir, 'data-set')):
        choice = input("Would you like to create a 'data-set' folder in '%s'?" %root_dir).lower()
        if choice in yes:
            save_dir = os.path.join(root_dir, 'data-set')
            os.mkdir(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'train')):
                os.mkdir(os.path.join(save_dir, 'train'))
            if not os.path.exists(os.path.join(save_dir, 'test')):
                os.mkdir(os.path.join(save_dir, 'test'))
        else:
            raise Exception('Data-set directory was not created, no data was extracted.')

    elif os.path.exists(out_dir):
        save_dir = out_dir
        if not os.path.exists(os.path.join(save_dir, 'train')):
            os.mkdir(os.path.join(save_dir, 'train'))
        if not os.path.exists(os.path.join(save_dir, 'test')):
            os.mkdir(os.path.join(save_dir, 'test'))

    else:
        choice = input('Specified output directory not found, would you like to create it? (y)/(n)')
        if choice in yes:
            os.mkdir(out_dir)
            os.mkdir(os.path.join(out_dir, 'train'))
            os.mkdir(os.path.join(out_dir, 'test'))
            save_dir = out_dir
        else:
            raise Exception('Specified save directory not found, and not created')

    print('Saving dataset to:', save_dir)

    # Extracting data-points from excel file:
    data_points = extract_excel_data(project_dir, classes=classes)

    # Create training and test data-sets:
    for mode, dirs in zip(['train', 'test'], [train_data_dirs, test_data_dirs]):

        output_dir = os.path.join(save_dir, mode, project)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for video_dir in dirs:
            print(video_dir)
            extract_all_frames(video_dir, data_points, output_dir, delay=delay, channels=channels)

    print('Dataset created for project', project)


if __name__ == "__main__":
    create_dataset('LingShui', delay=2.600,
                   root_dir=r'C:\Users\MTN\PycharmProjects\Survey_anomaly_detection\pycharm\Anomaly_detection\data')
