from locate.locate import locate 
from video.inputconfig import inputconfig
from track.track import track
from locate.defuse import defuse,invdefuse
from video.result import video,result, videocomp


input_folder = "input/yourdatafolder"

# This function will create .png files in appropriate folder from the videos in the input folder
frame = inputconfig(input_folder)

# This function will create a output folder who will contain the folder list_comp containing all the detected macrophages in each frame and list_sep which conain a .png for each macrophage in each frames
locate(input_folder)

# This function will separate the macrophage in frame i if they touch in i and not in i-1
defuse()

# This function will create the folder list_track in output that will contain for each macrophage the frame where they're shown
# The parameters is the threshold for the Intersection over Union needed 
track(0.5)

# This function create two videos result_video.mp4 and result_video_v.mp4 with all macrophage tracked and located 
result(input_folder)
video()
