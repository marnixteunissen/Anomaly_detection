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
    columns = ['Date', 'Time', 'Primary Code', 'Secondary Code']
    # Opening the Datasheet as DateFrame(s)
    excel_data = pd.read_excel(excel_file, sheet_name=classes, usecols=columns)

    if project.endswith('Turkstream'):
        excel_data['Date'] = [d.date() for d in excel_data['Date']]

    excel_data['Time'] = [pd.to_datetime(d, format=' %H:%M:%S.%f').time() if type(d) == str and d.startswith(' ')
                          else pd.to_datetime(d, format='%H:%M:%S.%f').time() if type(d) == str else d
                          for d in excel_data['Time']]

    excel_data['Secondary Code'] = ['FJ' + code if len(code) < 4 else code for code in excel_data['Secondary Code']]

    # Add column with date and time concatenated:
    excel_data['datetime'] = excel_data.apply(lambda r: datetime.combine(r['Date'], r['Time']), 1)
    return excel_data


def extract_video_events(excel_data, video, static_offset=0.000):
    '''
    Extracts the timestamps with labels from the events sheet.
    :param excel_data:      Pandas DataFrame (not dict of DataFrames)
                            Data extracted from the excel sheet with the
                            extract_excel_data function.
    :param video:           Path string
                            Path to folder where video feeds are located
    :param static_offset:   float
                            offset in seconds
    :return:                Pandas Dataframe
                            containing only timestamp (string) and Secondary Code (string)
    '''

    # Reading timestamp from directory name
    if video.endswith('.mp4'):
        file = True
        time_string = os.path.split(video)[-1].split('@')[0].split()[0]
    elif video.split('_')[-1].isnumeric():
        time_string = os.path.split(video)[-1].split('DATA_')[-1]
        file = False
    else:
        print('Check source input, time string not found in filename')
        raise ReferenceError('File path not consistent with project structure')

    # Extract the timestamp of the start and end of the video from the filename:
    first_stamp = pd.to_datetime(time_string, format="%Y%m%d%H%M%S%f")
    if file:
        video_file = video
    else:
        video_file = [x for x in os.listdir(video) if x.endswith('.mp4')][0]

    cap = cv2.VideoCapture(os.path.join(video, video_file))
    assert(cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_sec = float(total_frames) / float(fps)
    last_stamp = (first_stamp + timedelta(seconds=total_sec))

    # Create pandas DataFrame with DateTime and labels from Secondary Code
    # for all events in excel_data:
    timestamps = excel_data[((first_stamp <= excel_data['datetime']) & (excel_data['datetime'] <= last_stamp))][['datetime', 'code']]
    # Calculating elapsed time in video
    offset = timedelta(seconds=static_offset)
    timestamps['Video Stamp'] = (timestamps['datetime'] - first_stamp + offset)
    timestamps['ms in video'] = (timestamps['Video Stamp'].dt.total_seconds() * 1000).astype('int')

    data_points = timestamps[['datetime', 'Video Stamp', 'ms in video', 'code']]

    return data_points


def extract_all_events(project_dir):
    # Find excel sheet in project folder:
    dir_files = os.listdir(project_dir)
    excel_files = [os.path.join(project_dir, file) for file in dir_files if file.endswith('.xlsx')]
    assert len(excel_files) == 1, "{} available excel files were found, " \
                                  "make sure one single excel file is available in the project directory".format(len(excel_files))
    # TODO: ask for user input if len(excel_files) != 0
    excel_file = excel_files[0]

    xlsx = pd.ExcelFile(excel_file)
    event_sheets = [name for name in xlsx.sheet_names if (name.lower().count('event') != 0
                                                          or name.lower().count('observation') != 0)]

    assert len(event_sheets) == 1
    # TODO: user selection of appropriate sheet with all events if assertion fails
    event_sheet = event_sheets[0]

    if project_dir.endswith('Troll'):
        columns_of_interest = ['date', 'time', 'secondarycode']
    else:
        columns_of_interest = ['date', 'time', 'primarycode', 'secondarycode']

    # Open event sheet:
    print('Opening excel sheet with events...')
    full_event_sheet = pd.read_excel(excel_file, sheet_name=event_sheet, dtype=str)

    # Copy the relevant columns to new dataframe:
    columns_to_copy = []
    original_columns = full_event_sheet.keys()
    clean_columns = [col.replace(" ", "").replace("\n", "").lower() for col in original_columns]

    for col in columns_of_interest:
        columns_to_copy.extend([column for column in clean_columns if column.startswith(col)])

    full_event_sheet.columns = clean_columns
    events = full_event_sheet[columns_to_copy]

    # renaming columns:
    events.columns = columns_of_interest

    for col in events.keys()[:2]:
        events[col] = pd.to_datetime(events[col], infer_datetime_format=True, errors='coerce')
        events.dropna(subset=[col], inplace=True)

    events['datetime'] = events.apply(lambda r: datetime.combine(r['date'].date(), r['time'].time()), 1)

    if project_dir.endswith('Troll'):
        events['code'] = events['secondarycode']
    else:
        events.loc[:, 'code'] = events.loc[:, 'primarycode'] + events.loc[:, 'secondarycode']

    return events[['datetime', 'code']]


def select_events(events, event_types=['FJOK']):
    selected_events = events.loc[events['code'].isin(event_types)]
    return selected_events


def get_event_types(project_dir, event_types=['FJOK']):
    events = extract_all_events(project_dir)
    return select_events(events, event_types)


if __name__ == "__main__":
    #dir = os.path.join(os.getcwd(), 'data')
    dir = r'C:\Users\MTN\Documents\Anomaly_Detection\Data'
    print('Working Dir:', dir)
    projects = os.listdir(dir)
    print(projects)
    for project in ['Tulip Oil']:
        print('converting project {}'.format(project))
        proj_dir = os.path.join(dir, project)
        eventing_list = get_event_types(proj_dir, ['FJOK'])
        #eventing_list.to_csv(os.path.join(proj_dir, 'Events {}.csv'.format(project)), index=False)
        print('project converted.')

    vid_events = extract_video_events(eventing_list, r'C:\Users\MTN\Documents\Anomaly_Detection\Data\Tulip Oil\Video\DATA_20180620122949270', static_offset=0.000)
    print(vid_events)
