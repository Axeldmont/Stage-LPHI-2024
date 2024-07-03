import os
from moviepy.editor import VideoFileClip

def convert_avi_to_mp4(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".avi"):
            avi_path = os.path.join(directory, filename)
            mp4_path = os.path.join(directory, filename[:-4] + ".mp4")
            clip = VideoFileClip(avi_path)
            clip.write_videofile(mp4_path, codec="libx264")
            clip.close()

dossier = "input/convert"
convert_avi_to_mp4(dossier)
