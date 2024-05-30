import os
import shutil
import cv2
import numpy as np

def calculate_iou(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def find_containing_folder(image_name, root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path) and image_name in os.listdir(folder_path):
            return folder_path
    return None

def track(threshold_iou):
    c = 1
    output_track = "output/list_track"
    os.makedirs(output_track, exist_ok=True)

    heatmap_base = "output/list_sep/heatmap_test_"
    current_files = os.listdir(heatmap_base + "0")
    
    for filename in current_files:
        image_number = filename.split("_")[1].split(".")[0]
        image_path = os.path.join(heatmap_base + "0", filename)
        output_folder = os.path.join(output_track, f"macrophage_{c}")
        os.makedirs(output_folder, exist_ok=True)
        new_filename = f"0_{image_number}.png"
        output_path = os.path.join(output_folder, new_filename)
        shutil.copy(image_path, output_path)
        c = c + 1

    folder_dict = {}
    for i in range(1, 130):
        input_folder_i = heatmap_base + str(i)
        files_i = os.listdir(input_folder_i)

        prev_folder_files = {}
        prev_folder_path = heatmap_base + str(i-1)
        files_k = os.listdir(prev_folder_path)

        for k, file_k in enumerate(files_k):
            imagecomp_path = os.path.join(prev_folder_path, file_k)
            imagecomp = cv2.imread(imagecomp_path, cv2.IMREAD_GRAYSCALE)
            _, imagecomp_bin = cv2.threshold(imagecomp, 127, 255, cv2.THRESH_BINARY)
            prev_folder_files[k] = imagecomp_bin

        for j, file_i in enumerate(files_i):
            image_path = os.path.join(input_folder_i, file_i)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, image_bin = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            new_filename = f"{i}_{j}.png"
            matched = False

            for k, imagecomp_bin in prev_folder_files.items():
                iou = calculate_iou(image_bin, imagecomp_bin)
                previous_image_name = f"{i-1}_{k}.png"
                if iou > threshold_iou:
                    if previous_image_name not in folder_dict:
                        containing_folder = find_containing_folder(previous_image_name, output_track)
                        folder_dict[previous_image_name] = containing_folder
                    output_path = os.path.join(folder_dict[previous_image_name], new_filename)
                    shutil.copy(image_path, output_path)
                    matched = True
                    break

            if not matched:
                output_folder = os.path.join(output_track, f"macrophage_{c}")
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, new_filename)
                shutil.copy(image_path, output_path)
                folder_dict[new_filename] = output_folder
                c = c + 1

