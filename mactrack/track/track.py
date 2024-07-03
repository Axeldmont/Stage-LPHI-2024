import os
import shutil
import cv2
import numpy as np

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()
    return intersection / union if union != 0 else 0


def find_containing_folder(image_path, root_folder):
    image_name = os.path.basename(image_path)
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            if image_name in os.listdir(folder_path):
                return folder_path
    return None

def track(n, threshold_iou, image_storage):
    c = 1

    for j, filename in enumerate(os.listdir("output/list_def/heatmap_test_0")):
        image_number = filename.split("_")[1].split(".")[0]       
        image_path = os.path.join("output/list_def/heatmap_test_0", filename)
        output_folder = os.path.join("output/list_track", f"macrophage_{c}")
        os.makedirs(output_folder, exist_ok=True)
        new_filename = f"0_{image_number}.png"
        output_path = os.path.join(output_folder, new_filename)
        shutil.copy(image_path, output_path)
        c = c + 1

    for i in range (1, n):
        print(f"heatmap_test_{i}")
        current = len(image_storage.get_list(f"heatmap_test_{i}"))
        for j in range(0, current):
            image = image_storage.get_image(f"heatmap_test_{i}", f"object_{j}.png")
            image = image.toarray()
            _, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            previous = len(image_storage.get_list(f"heatmap_test_{i-1}"))

            for k in range(0, previous):
                imagecomp = image_storage.get_image(f"heatmap_test_{i-1}", f"object_{k}.png")
                imagecomp = imagecomp.toarray()
                _, image2 = cv2.threshold(imagecomp, 127, 255, cv2.THRESH_BINARY)

                iou = calculate_iou(image1, image2)
                containing_folder = find_containing_folder(f"output/list_track/{i-1}_{k}.png", "output/list_track")

                if iou > threshold_iou:
                    new_filename = f"{i}_{j}.png"
                    output_path = os.path.join(containing_folder, new_filename)
                    shutil.copy(f"output/list_def/heatmap_test_{i}/object_{j}.png", output_path)

            folder = find_containing_folder(f"output/list_track/{i}_{j}.png", "output/list_track")
            if folder == None :
                    output_folder = os.path.join("output/list_track", f"macrophage_{c}")
                    os.makedirs(output_folder, exist_ok=True)
                    new_filename = f"{i}_{j}.png"
                    output_path = os.path.join(output_folder, new_filename)
                    shutil.copy(f"output/list_def/heatmap_test_{i}/object_{j}.png", output_path)
                    c = c + 1
