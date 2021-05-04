from Anomaly_detection.file_functions import *
from Anomaly_detection.excel_functions import *
import os
import subprocess

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
		subprocess.Popen(cmd)


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
	video_files = get_video_file_names(video_dir, channels)
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
			subprocess.Popen(cmd)
		print('Keyframes are being extracted for file ', video_file)


if __name__ == "__main__":
	current_dir = os.getcwd()
	video_dir = os.path.join(current_dir, '\\data\\DATA_20200423153202169')
	excel = extract_excel_data((current_dir + '\\data'), classes='Field Joint')

	sample_data = extract_video_events(excel, video_dir)
	extract_batch_frames(video_dir, sample_data, out_dir=(current_dir + '\data\Samples'), channels=[2])
