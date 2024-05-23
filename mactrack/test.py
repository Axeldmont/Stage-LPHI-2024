from locate.locate import locate 
from inputconfig import inputconfig
from track.tracktest import track
#from locate.defuse import defuse
import time

start_time = time.time()
input_folder = "input/fish2_mp4"
#inputconfig(input_folder)
locate(input_folder)
# defuse()
#track(0.5)

end_time = time.time()  
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds TOTAL")