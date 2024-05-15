from locate.locate import locate 
from inputconfig import inputconfig
from track.track import track

input_folder = "input/fish2_mp4"
inputconfig(input_folder)
locate(input_folder)
track(0.5)