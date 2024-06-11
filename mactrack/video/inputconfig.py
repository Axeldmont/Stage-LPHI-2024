import cv2
import os
from video.frame import VideoFrames

def inputconfig(input_folder):
    input_folder_v = os.path.join(input_folder,"vert")
    output_folder = os.path.join(input_folder, "dataset/test/test_x")
    output_folder_v = os.path.join(input_folder_v, "frames")
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    if len(video_files) != 1:
        print("Erreur: Aucun fichier vidéo ou plusieurs fichiers vidéo trouvés dans le dossier.")
        return
    
    video_files_v = [f for f in os.listdir(input_folder_v) if f.endswith(".mp4")]
    
    if len(video_files_v) != 1:
        print("Erreur: Aucun fichier vidéo ou plusieurs fichiers vidéo trouvés dans le dossier.")
        return

    video_filename = video_files[0]
    video_path = os.path.join(input_folder, video_filename)
    video_filename_v = video_files_v[0]
    video_path_v = os.path.join(input_folder_v, video_filename_v)

    video_capture = cv2.VideoCapture(video_path)
    video_capture_v = cv2.VideoCapture(video_path_v)
    count = 0
    video_frames = VideoFrames()

    while True:
        success, frame = video_capture.read()
        if not success:
            break        
        success_v, frame_v = video_capture_v.read()
        if not success:
            break

        video_frames.add_frame(frame)
        video_frames.add_frame_v(frame_v)
        filename = os.path.join(output_folder, f"{count:03d}_image.png")
        cv2.imwrite(filename, frame)
        filename_v = os.path.join(output_folder_v, f"{count:03d}_image.png")
        cv2.imwrite(filename_v, frame_v)
        count += 1
    
    video_capture.release()
    return video_frames


