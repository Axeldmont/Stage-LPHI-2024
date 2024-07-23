from locate.locate import locate 
from locate.list_sep import segmentation
from locate.defuse import defuse, invdefuse
from track.track import track
from video.inputconfig import inputconfig
from video.result import video,result, videocomp
import os
from analyse.intensity import intensity,intensitymed
from analyse.distance import distance
from analyse.size import size
from analyse.perimeter import perimeter
from analyse.recap import aggregate
import shutil
from track.filtre import supprimer_petit

input_folder = "input/yourdatafolder" # Enter your data folder name instead of yourdatafolder
n = 130 # Number of frame in the initial video 
p = 10 # minimal number of frame where you can track your macrophage, if it is present in less or equal p frame it will not be tracked

# create each frame of the video into picture folder (red picture are stored in "input/yourdatafolder/dataset/test/test_x", green picture are stored in "input/yourdatafolder/vert/frames")
frame = inputconfig(input_folder)

# Function which will do the segmentation, it will create a folder output with in it "list_comp" which contain the segmentation at each frame and "list_sep" which contain every object in separate picture file at each frame 
locate(input_folder)

# This will stock all the picture in class in order to accelerate the program
image_storage = segmentation("output/list_sep")
image_storage.load_images()

# This will prevent and separate the merging macrophage 
image_storage = defuse(n, image_storage)
image_storage = invdefuse(n, image_storage)

# This will do the tracking of every macrophage and filter if you detect a macrophage on enough frame
track(n, 0.5, image_storage)
supprimer_petit(p)

# This will create in the output folder two video which show the results of the tracking
result(input_folder)
video()

# This is for deleting non necessary folder to liberate some place 
shutil.rmtree("output/list_def")
shutil.rmtree("output/list_sep")
shutil.rmtree("output/result")
shutil.rmtree("output/resultv")

# This is to create folder who will contain the data 
if not os.path.exists("output/data"):
    os.makedirs("output/data")
if not os.path.exists("output/plot"):
    os.makedirs("output/plot")

# These will collect the data and stock them in dataframe 
intmed = intensitymed(n,frame, input_folder) # Data on the intensity of macrophage 
dis = distance(n) # data of distance to the right border
siz = size(n) # data of the size of the macrophage 
per = perimeter(n) # data of the perimeter of the macrophage 
recap = aggregate(dis,intmed,siz,per) # summary of each data for each macrophage in a movie 
