import cv2
import os
from video.frame import VideoFrames

def inputconfig(input_folder):
    output_folder = os.path.join(input_folder, "dataset/test/test_x")
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    if len(video_files) != 1:
        print("Erreur: Aucun fichier vidéo ou plusieurs fichiers vidéo trouvés dans le dossier.")
        return
    
    video_filename = video_files[0]
    video_path = os.path.join(input_folder, video_filename)

    video_capture = cv2.VideoCapture(video_path)
    count = 0
    video_frames = VideoFrames()

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        video_frames.add_frame(frame)
        filename = os.path.join(output_folder, f"{count:03d}_image.png")
        cv2.imwrite(filename, frame)
        count += 1
    
    video_capture.release()
    return video_frames

def inputconfigv(input_folder):
    input_folder_v = os.path.join(input_folder,"vert")
    output_folder = os.path.join(input_folder_v, "frames")
    video_files = [f for f in os.listdir(input_folder_v) if f.endswith(".mp4")]
    
    if len(video_files) != 1:
        print("Erreur: Aucun fichier vidéo ou plusieurs fichiers vidéo trouvés dans le dossier.")
        return
    
    video_filename = video_files[0]
    video_path = os.path.join(input_folder_v, video_filename)

    video_capture = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        filename = os.path.join(output_folder, f"{count:03d}_image.png")
        cv2.imwrite(filename, frame)
        count += 1
    
    video_capture.release()


