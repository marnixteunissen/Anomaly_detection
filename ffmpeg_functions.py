from file_functions import *
from excel_functions import *
import os
import subprocess
import sys, getopt

# ffmpeg -i inFile -f image2 -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr oString%03d.png
# ffmpeg -i yosemiteA.mp4 -ss 00:00:18.123 -frames:v 1 yosemite.png


def extract_frame(video_file, time_stamp, event_nr, code='FJOK', out_dir=''):
    """
    Function to extract single frame from video file.
    Creates a snapshot of video at the event-time as a
    .png file in the current working directory.
    filename contains timestamp and event type


    :param video_file: string, filepath to original video
    :param time_stamp: string, timestamp of event
    :param event_nr: int, index in excel datasheet
    :param code: string, default: 'FJOK'. type of event.
    :param out_dir: string, default: ''. Path to save outputs to.
    """
    working_dir = os.getcwd()
    if out_dir == '':
        save_dir = working_dir
    elif os.path.exists(out_dir):
        save_dir = out_dir
    else:
        raise FileNotFoundError('Directory ', out_dir, ' was not found.')
    print('save_dir = ', save_dir)
    infile = video_file
    print('video file = ', video_file)
    ffmpeg = working_dir + '\\bin\\ffmpeg'
    #outfile = save_dir+ '\\' + code + str(event_nr) + '.png'
    #print('outfile = ', outfile)
    # command for single frame looks like:
    # ffmpeg -i <input_file> -ss <timestamp(hh:mm:ss:000)> -frames:v 1 <output_file.png>
    for (time, event) in zip(time_stamp, event_nr):
        outfile = save_dir + '\\' + code + str(event) + '.png'
        print('outfile = ', outfile)
        cmd = 'ffmpeg' + ' -y ' + '-i ' + infile + ' -ss ' + time + ' -frames:v 1 ' + outfile
        print(cmd)
        # pass command:
        subprocess.run(cmd)


def extract_batch_frames(video_dir, sample_data, out_dir='', channels=[2]):
    """
    Creates .png images for the keyframes given in the DataFrame sample_data, which should be created with
    the extract_video_events() function
    :param video_dir: 	Path (string)
                        Path to the directory where the video files are located
    :param sample_data:	Dataframe
                        Contains sample nr as index, timestamp of event and type of event (code)
    :param out_dir: 	Path (string)
                        Directory where the .png files should be saved, should be an existing directory
    :param channels: 	int, list of int
                        Channels from which the .png files should be extracted
    """
    video_files = get_video_file_names(video_dir, channels, )
    working_dir = os.getcwd()
    if out_dir == '':
        save_dir = working_dir
    elif os.path.exists(out_dir):
        save_dir = out_dir
    else:
        raise FileNotFoundError('Directory ', out_dir, ' was not found, please use existing directory')

    # creating lists for iterating:
    codes = sample_data['Secondary Code'].to_list()
    sample_nrs = sample_data.index.to_list()
    time_stamps = sample_data['Video Stamp (string)'].to_list()

    # Iterate over channels:
    for video_file in video_files:
        infile = os.path.join(video_dir, video_file)
        # Spawn parallel subprocesses for saving files:
        for (sample, code, time_stamp) in zip(sample_nrs, codes, time_stamps):
            outfile = save_dir + '\\' + code + str(sample) + '.png'
            cmd = 'ffmpeg' + ' -y ' + '-i ' + infile + ' -ss ' + time_stamp + ' -frames:v 1 ' + outfile
            print(cmd)
            subprocess.run(cmd, shell=True)

        print('Keyframes were extracted for file ', video_file)


def convert_to_mp4(input_file, dry_run=False):
    """
    Converts a file to mp4. Requires ffmpeg and libx264
    input_file -- The file to convert
    dry_run -- Whether to actually convert the file
    """
    output_file = input_file + '.mp4'
    ffmpeg_command = 'ffmpeg -loglevel quiet -i "%s" -vcodec libx264 -b 700k -s 480x368 -acodec libfaac -ab 128k -ar 48000 -f mp4 -deinterlace -y -threads 4 "%s" ' % (input_file,output_file)

    if not os.path.exists(output_file):
        if not os.path.exists(input_file):
            print("%s was queued, but does not exist" % input_file)
            return

        if dry_run:
            print("%s" % input_file)
            return

        print("Converting %s to MP4\n" % input_file)

        # ffmpeg
        print(subprocess.call(ffmpeg_command,shell=True))

        # qtfaststart so it streams
        print (subprocess.call('qtfaststart "%s"' % output_file,shell=True))

        # permission fix
        print(subprocess.call('chmod 777 "%s"' % output_file,shell=True))

        print("Done.\n\n")

    elif not dry_run:
        print("%s already exists. Aborting conversion." % output_file)


def convert_all_to_mp4(input_dir):
    """
    Converts all files in a folder to mp4
    input_dir -- The directory in which to look for files to convert
    allowed_extensions -- The file types to convert to mp4
    dry_run -- If set to True, only outputs the file names
    """

    for root, dirs, files in os.walk(input_dir):
        for name in files:
            if name.lower().endswith('ch2.asf'):
                file = os.path.join(root, name)
                output = file.split('.')[0] + '.mp4'
                cmd = 'ffmpeg -i "{}" -c:v libx264 -strict -2 "{}"'.format(file, output)
                subprocess.run(cmd)


if __name__ == "__main__":
    input_dir = r'C:\Users\MTN\Documents\Anomaly_Detection\Data\Noble Leviathan\GL-3'
    convert_all_to_mp4(input_dir)


# current_dir = os.getcwd()
    # print(current_dir)
    # video_dir = (current_dir + '\\data\\DATA_20200423203210192')
    # print(video_dir)
    # excel = extract_excel_data((current_dir + '\\data'), classes='Field Joint')
#
    # sample_data = extract_video_events(excel, video_dir, -1.992)
#
    # extract_batch_frames(video_dir, sample_data, out_dir=(current_dir + '\data\Samples'), channels=[2])
