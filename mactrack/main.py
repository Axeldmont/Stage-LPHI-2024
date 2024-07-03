from locate.locate import locate 
from locate.list_sep import segmentation
from video.inputconfig import inputconfig
from track.track import track
from locate.defuse import defuse,invdefuse
from video.result import video,result, videocomp
import time
from line_profiler import LineProfiler
import cv2
import numpy as np
import os
from analyse.intensity import intensity,intensitymed
from analyse.distance import distance
from analyse.size import size
from analyse.perimeter import perimeter
from analyse.recap import aggregate

start_time = time.time()
input_folder = "input/3hpa_fish2"
n = 130
if not os.path.exists("output/data"):
    os.makedirs("output/data")
if not os.path.exists("output/plot"):
    os.makedirs("output/plot")
frame = inputconfig(input_folder)
int = intensity(n, frame, input_folder)
intmed = intensitymed(n,frame, input_folder)
dis = distance(n)
siz = size(n)
per = perimeter(n)
recap = aggregate(dis,int,intmed,siz,per)

#locate(input_folder)
#image_storage = segmentation("output/list_sep")
#image_storage.load_images()

#image_storage = defuse(n, image_storage)
#invdefuse() pas encore bon correctif sur process 

#track(n, 0.5, image_storage)

#result(input_folder)
#video()
#videocomp()

########################################################
#profiler = LineProfiler()
#profiler.add_function(defuse)
#profiler.run('defuse(image_storage)')
#profiler.print_stats()


end_time = time.time()  
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds TOTAL")








