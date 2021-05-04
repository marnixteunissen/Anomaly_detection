import pandas as pd
import os
from datetime import datetime, timedelta


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
    columns = ['Date', 'Time', 'KP(km)', 'KP meter', 'Primary Code', 'Secondary Code']
    # Opening the Datasheet as DateFrame(s)
    excel_data = pd.read_excel(excel_file, sheet_name=classes, usecols=columns)
    # Add column with date and time concatenated:
    excel_data['datetime'] = excel_data.apply(lambda r: datetime.combine(r['Date'], r['Time']), 1)

    return excel_data


def extract_video_events(excel_data, video_folder, static_offset=timedelta(seconds=0)):
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
    last_stamp = first_stamp + timedelta(minutes=30)

    # Create pandas DataFrame with DateTime and labels from Secondary Code
    # for all events in excel_data:
    timestamps = excel_data[(first_stamp <= excel_data['datetime']) &
                            (excel_data['datetime'] <= last_stamp)][['datetime', 'Secondary Code']]
    # Calculating elapsed time in video
    offset = timedelta(seconds=static_offset)
    timestamps['Video Stamp'] = timestamps['datetime'] - first_stamp + offset
    timestamps['Video Stamp (string)'] = timestamps['Video Stamp'].astype('string').str.split().str[-1]
    print(timestamps)
    data_points = timestamps[['Video Stamp (string)', 'Secondary Code']]

    return data_points


if __name__ == "__main__":
    excel = extract_excel_data('C:\\Users\\MTN\\Documents\\Survey_anomaly_detection\\pycharm\\Anomaly_detection\\data')
    print(excel)
    time_stamps = extract_video_events(excel, 'C:\\Users\\MTN\\Documents\\Survey_anomaly_detection\\'
                                              'pycharm\\Anomaly_detection\\data\\DATA_20200423153202169', 1.992)
    print(time_stamps)

