# Setting up:
### 1. Downloading and installing ffmpeg.
Following the tuorial from:
https://www.thewindowsclub.com/how-to-install-ffmpeg-on-windows-10#:~:text=Navigate%20to%20Advanced%20button%20and,bin%5C%E2%80%9D%20and%20click%20OK.

The file can be extracted to the same folder as the pycharm project is in. Important: adding ffmpeg to path.



#### extracting frames:
https://www.bogotobogo.com/FFMpeg/ffmpeg_thumbnails_select_scene_iframe.php
#### metadata from file:
http://ffmpeg.org/ffmpeg-formats.html#Metadata-1
#### executing shell commands in python:
https://janakiev.com/blog/python-shell-commands/

#### Possible alternatives:
Instead of ffmpeg, using openCV or equivalent

#### Possible problems:
ffmpeg not working when trying to extract frames with python functions from server files.

Downloading all files could be time consuming.

Extracting metadata using ffmpeg has to be improved, shows enough but 
outputs too little to .txt file

possibly incoming video files have to be extracted frame by frame for classification 
--> look for ways touse mp4 as input, or webcam-like input.

