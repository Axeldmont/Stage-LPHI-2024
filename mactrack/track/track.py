import os
import shutil
import cv2
import numpy as np

def calculate_iou(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def find_containing_folder(image_path, root_folder):
    image_name = os.path.basename(image_path)
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            if image_name in os.listdir(folder_path):
                return folder_path
    return None

def track(threshold_iou):

    c = 1

    for j, filename in enumerate(os.listdir("output/list_sep/heatmap_test_0")):
        image_number = filename.split("_")[1].split(".")[0]       
        image_path = os.path.join("output/list_sep/heatmap_test_0", filename)
        output_folder = os.path.join("output/list_track", f"macrophage_{c}")
        os.makedirs(output_folder, exist_ok=True)
        new_filename = f"0_{image_number}.png"
        output_path = os.path.join(output_folder, new_filename)
        shutil.copy(image_path, output_path)
        c += 1

    for i in range (1,130):
        input_folder_i = f"output/list_sep/heatmap_test_{i}"
        files_i = os.listdir(input_folder_i)

        for j in range(0,len(files_i)):
            image_path = os.path.join(input_folder_i, f"object_{j}.png")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

            input_folder_k = f"output/list_sep/heatmap_test_{i-1}"
            files_k = os.listdir(input_folder_k)
            for k in range(0,len(files_k)):
                imagecomp_path = os.path.join(input_folder_k, f"object_{k}.png")
                imagecomp = cv2.imread(imagecomp_path, cv2.IMREAD_GRAYSCALE)
                _, image2 = cv2.threshold(imagecomp, 127, 255, cv2.THRESH_BINARY)

                iou = calculate_iou(image1, image2)
                containing_folder = find_containing_folder(f"output/list_track/{i-1}_{k}.png", "output/list_track")

                if iou > threshold_iou:
                    new_filename = f"{i}_{j}.png"
                    output_path = os.path.join(containing_folder, new_filename)
                    shutil.copy(image_path, output_path)

            folder = find_containing_folder(f"output/list_track/{i}_{j}.png", "output/list_track")
            if folder == None :
                    output_folder = os.path.join("output/list_track", f"macrophage_{c}")
                    os.makedirs(output_folder, exist_ok=True)
                    new_filename = f"{i}_{j}.png"
                    output_path = os.path.join(output_folder, new_filename)
                    shutil.copy(image_path, output_path)
                    c += 1
