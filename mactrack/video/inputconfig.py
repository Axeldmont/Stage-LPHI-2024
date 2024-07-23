import cv2
import os
from video.frame import VideoFrames
import numpy as np

def median_green_intensity(input_folder, output_path):
    image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

    if not image_files:
        print("No image files found in the folder.")
        return

    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path, cv2.IMREAD_COLOR)
    height, width, _ = first_image.shape

    green_channel_values = np.zeros((height, width, len(image_files)), dtype=np.uint8)

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        green_channel_values[:, :, idx] = image[:, :, 1]

    median_green = np.median(green_channel_values, axis=2).astype(np.uint8)
    median_green_image = np.zeros((height, width, 3), dtype=np.uint8)
    median_green_image[:, :, 1] = median_green 
    output_file = os.path.join(output_path, 'mediane.png')
    cv2.imwrite(output_file, median_green_image)

def create_average_green_image(folder_path, output_path):
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not image_files:
        raise ValueError("No images found in the folder.")
    green_sum = None
    image_count = 0
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {image_file}, skipping.")
            continue

        if green_sum is None:
            green_sum = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64)

        if image.shape[:2] != green_sum.shape:
            raise ValueError(f"Image size mismatch: {image_file} has different dimensions.")

        green_sum += image[:, :, 1]
        image_count += 1
    
    if image_count == 0:
        raise ValueError("No valid images found in the folder.")

    green_mean = (green_sum / image_count).astype(np.uint8)
    average_green_image = np.zeros((green_mean.shape[0], green_mean.shape[1], 3), dtype=np.uint8)
    average_green_image[:, :, 1] = green_mean
    cv2.imwrite(os.path.join(output_path,"moyenne.png"), average_green_image)
    print(f"Average green intensity image saved to {output_path}")

def inputconfig(input_folder):
    input_folder_v = os.path.join(input_folder, "vert")
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
        if not success_v:
            break

        frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame_v_resized = cv2.resize(frame_v, (frame_v.shape[1] // 2, frame_v.shape[0] // 2))

        video_frames.add_frame(frame_resized)
        video_frames.add_frame_v(frame_v_resized)

        filename = os.path.join(output_folder, f"{count:03d}_image.png")
        cv2.imwrite(filename, frame_resized)
        filename_v = os.path.join(output_folder_v, f"{count:03d}_image.png")
        cv2.imwrite(filename_v, frame_v_resized)
        count += 1
    
    video_capture.release()
    video_capture_v.release()
    create_average_green_image(output_folder_v, input_folder_v)
    median_green_intensity(output_folder_v, input_folder_v)

    return video_frames


