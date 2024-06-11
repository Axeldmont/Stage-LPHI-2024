from locate.locate import locate 
from video.inputconfig import inputconfig
from track.track import track
from locate.defuse import defuse,invdefuse
from video.result import video,result, videocomp
import time
from line_profiler import LineProfiler
import cv2

start_time = time.time()
input_folder = "input/fish3_mp4"
frame = inputconfig(input_folder)
#locate(input_folder)
#defuse()
#invdefuse() pas encore bon correctif sur process 
#track(0.5)
#result(input_folder)
#video()
#videocomp()

########################################################
#profiler = LineProfiler()
#profiler.add_function(video)
#profiler.run('video()')
#profiler.print_stats()


end_time = time.time()  
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds TOTAL")








