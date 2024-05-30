from locate.locate import locate 
from inputconfig import inputconfig
from track.track import track
from locate.defuse2 import defuse,invdefuse
import time

start_time = time.time()
input_folder = "input/fish3_mp4"
#inputconfig(input_folder)
#locate(input_folder)
#defuse()
#invdefuse() pas encore bon correctif sur process 
#track(0.5)



########################################################

import cv2
import numpy as np
import os

# Dossier contenant les dossiers d'images segmentées
segmented_images_root = 'output/list_track'
# Chemin de la vidéo source
video_path = 'input/fish3_mp4/fish3.mp4'
# Dossier de sortie pour la nouvelle vidéo
output_dir = 'output'
output_path = os.path.join(output_dir, 'annotated_video.mp4')

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Lire la vidéo
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Lire toutes les images segmentées et les stocker dans un dictionnaire
segmented_images = {}
for folder_name in os.listdir(segmented_images_root):
    if folder_name.startswith("macrophage_"):
        object_id = folder_name.split("_")[1]
        folder_path = os.path.join(segmented_images_root, folder_name)
        for image_name in os.listdir(folder_path):
            if image_name.endswith(".png"):
                frame_number = int(image_name.split("_")[0])
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if frame_number not in segmented_images:
                    segmented_images[frame_number] = []
                segmented_images[frame_number].append((img, object_id))

# Parcourir les frames de la vidéo
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count in segmented_images:
        for mask, object_id in segmented_images[frame_count]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                # Trouver le centre du contour pour placer le numéro
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    cv2.putText(frame, object_id, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()


end_time = time.time()  
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds TOTAL")