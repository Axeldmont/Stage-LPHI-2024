import cv2
import os

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []

    def extract_frames(self, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
            self.frames.append(Frame(frame_filename))
            frame_count += 1

        cap.release()

class Frame:
    def __init__(self, image_path):
        self.image_path = image_path
        self.rois = []

    def add_roi(self, roi_folder):
        frame_name = os.path.basename(self.image_path)
        frame_number = int(frame_name.split('_')[1].split('.')[0])

        roi_folder_path = os.path.join(roi_folder, f"heatmap_test_{frame_number}")
        if not os.path.exists(roi_folder_path):
            print("fail")
            return 

        roi_files = [f for f in os.listdir(roi_folder_path) if f.endswith('.png')]
        for roi_file in roi_files:
            roi_path = os.path.join(roi_folder_path, roi_file)
            self.rois.append(ROI(roi_path))

class ROI:
    def __init__(self, roi_path):
        self.roi_path = roi_path
        self.macrophage = [] 

    def detect_macrophage(self):
        roi_image = cv2.imread(self.roi_path, cv2.IMREAD_GRAYSCALE)
        white_pixels = (roi_image > 200).sum()
        self.macrophage.append(white_pixels)

video_path = "fish2.mp4"
output_folder = "test"
roi_folder = "list" 

video = Video(video_path)
video.extract_frames(output_folder)

for frame in video.frames:
    frame.add_roi(roi_folder)

for frame in video.frames:
    for roi in frame.rois:
        roi.detect_macrophage()
        
with open("taille.txt", "w") as file:
    for frame in video.frames:
        for roi in frame.rois:
            file.write(f"Taille macrophage pour le ROI : {roi.roi_path}\n")
            file.write(str(roi.macrophage) + "\n")
