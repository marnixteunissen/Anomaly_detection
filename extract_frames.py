import sys
import getopt
import os
import subprocess

# ffmpeg -i inFile -f image2 -vf "select='eq(pict_type,PICT_TYPE_I)'" -vsync vfr oString%03d.png

def extract_frame(video_file, time_stamp):
	working_dir = os.getcwd()
	inFile = video_file
	ffmpeg = working_dir = + '\\bin\\ffmpeg'
	cmd = [ffmpeg, '-i', inFile, '-f', 'image2', '-vf',
		   "select='eq(pict_type,PICT_TYPE_I)'", '-vsync', 'vfr', outFile]

	# command for single frame looks like:
	# ffmpeg -i <input_file> -ss <timestamp(hh:mm:ss:000)> -frames:v 1 <output_file.png>

	# pass command:
	subprocess.call(cmd)



def main(argv):
	inFile = ''
	oString = 'out'
	usage = 'usage: python iframe.py -i <inputfile> [-o <oString>]'
	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","oString="])
	except getopt.GetoptError:
		print(usage)
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(usage)
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inFile = arg
		elif opt in ("-o", "--oString"):
			oString = arg
	print('Input file is "', inFile)
	print('oString is "', oString)

		# need input, otherwise exit
	if inFile == '':
		print(usage)
		sys.exit()

	# start extracting i-frames
	#home = os.path.expanduser("~")
	working_dir = os.getcwd()
	ffmpeg = working_dir + '\\bin\\ffmpeg'
	inFile = working_dir = inFile
	#ffmpeg = home + '\\bin\\ffmpeg'
	outFile = oString + '%03d.png'

	cmd = [ffmpeg,'-i', inFile,'-f', 'image2','-vf',
			   "select='eq(pict_type,PICT_TYPE_I)'",'-vsync','vfr',outFile]
	print(cmd)
	subprocess.call(cmd)

if __name__ == "__main__":
	main(sys.argv[1:])