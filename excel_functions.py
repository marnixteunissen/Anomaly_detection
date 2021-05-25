import pandas as pd
import os
from datetime import datetime, timedelta
import cv2


def extract_excel_data(project, classes='Field Joint'):
    '''
    Creates a pandas dataframe with data from the excel of the given project.
    In case of multiple classes, gives a dict of DataFrames (see pandas documentation for read_excel function)
    :param project:     Path (string)
                        Path to project folder where excel is located
    :param classes:     string or list, Default: 'Field Joint' L
                        ist of data labels to be extracted from excel file
    :return:            Pandas DataFrame or dict of DataFrames in case of a list of classes.
    '''
    # Finding the Datasheet:
    dir_files = os.listdir(project)
    for file in dir_files:
        if file.endswith(('.xlsx')):
            excel_file = os.path.join(project, file)

    # Select columns to import:
    columns = ['Date', 'Time', 'KP(km)', 'Primary Code', 'Secondary Code']
    # Opening the Datasheet as DateFrame(s)
    excel_data = pd.read_excel(excel_file, sheet_name=classes, usecols=columns)
    print(type(excel_data['Time'][0]))
    if type(excel_data['Time'][0]) != datetime:
        excel_data['Time'] = pd.to_datetime(excel_data['Time'])
    # Add column with date and time concatenated:
    excel_data['datetime'] = excel_data.apply(lambda r: datetime.combine(r['Date'], r['Time']), 1)

    return excel_data


def extract_video_events(excel_data, video_folder, static_offset=0.000):
    '''
    Extracts the timestamps with labels from the events sheet.
    :param excel_data:      Pandas DataFrame (not dict of DataFrames)
                            Data extracted from the excel sheet with the
                            extract_excel_data function, one label only.
    :param video_folder:    Path string
                            Path to folder where video feeds are located
    :param static_offset:   float
                            offset in seconds
    :return:                Pandas Dataframe
                            containing only timestamp (string) and Secondary Code (string)
    '''

    # Reading timestamp from directory name
    time_string = video_folder.split('DATA_')[-1]

    # Extract the timestamp of the start and end of the video from the filename:
    first_stamp = pd.to_datetime(time_string, format="%Y%m%d%H%M%S%f")

    video_file = [x for x in os.listdir(video_folder) if x.endswith('.mp4')][0]

    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = float(total_frames) / float(fps)
    last_stamp = first_stamp + timedelta(seconds=total_sec)

    # Create pandas DataFrame with DateTime and labels from Secondary Code
    # for all events in excel_data:
    timestamps = excel_data[(first_stamp <= excel_data['datetime']) &
                            (excel_data['datetime'] <= last_stamp)][['datetime', 'Secondary Code']]

    # Calculating elapsed time in video
    offset = timedelta(seconds=static_offset)
    timestamps['Video Stamp'] = (timestamps['datetime'] - first_stamp + offset)
    timestamps['ms in video'] = (timestamps['Video Stamp'].dt.total_seconds() * 1000).astype('int')

    data_points = timestamps[['Video Stamp', 'ms in video', 'Secondary Code']]

    return data_points


if __name__ == "__main__":
    dir = os.getcwd()
    print(dir)
    excel = extract_excel_data(dir + r'\data\Turkstream')
    #time_stamps = extract_video_events(excel, dir + r'\data\LingShui\Video\DATA_20200627074626222', -2.609)
    #print(time_stamps)




