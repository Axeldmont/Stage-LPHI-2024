import os
import cv2
import numpy as np
import shutil
from kartezio.inference import ModelPool
from kartezio.fitness import FitnessIOU
from kartezio.dataset import read_dataset
from numena.image.basics import image_normalize

def filter_small_shapes(image, min_size):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size:
            cv2.drawContours(image, [contour], 0, 0, -1) 
    
    return image

def extract_objects(image, output_dir, filename):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_dir = os.path.join(output_dir, filename.split('.')[0])
    os.makedirs(image_dir, exist_ok=True)
    object_count = 0

    for i, contour in enumerate(contours):
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, (255), -1)
        object_image = cv2.bitwise_and(image, image, mask=mask)
        cv2.imwrite(os.path.join(image_dir, f"object_{i}.png"), object_image)
        object_count += 1
    
    return object_count

def locate(input_folder):
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_dir_masks = os.path.join(output_dir, "masks")
    if not os.path.exists(output_dir_masks):
        os.makedirs(output_dir_masks)

    fitness = FitnessIOU()
    ensemble = ModelPool(os.path.join(input_folder, "models"), fitness, regex="*/elite.json").to_ensemble()
    dataset = read_dataset(os.path.join(input_folder, "dataset"), counting=True)
    p_test = ensemble.predict(dataset.test_x)

    for i in range(130):
        mask_list = [image_normalize(pi[0][i]["mask"]) for pi in p_test]
        heatmap = np.array(mask_list).mean(axis=0)
        heatmap_cp = (heatmap * 255.0).astype(np.uint8)

        cv2.imwrite(os.path.join(output_dir_masks, f"heatmap_test_{i}.png"), heatmap_cp)

    input_dir_masks = os.path.join(output_dir, "masks")
    output_dir_masks2 = os.path.join(output_dir, "list_comp")
    if not os.path.exists(output_dir_masks2):
        os.makedirs(output_dir_masks2)

    file_list = os.listdir(input_dir_masks)
    min_shape_size = 800 

    for filename in file_list:
        input_path = os.path.join(input_dir_masks, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        colored_image = 255 * (image > 0).astype('uint8')
        colored_image = filter_small_shapes(colored_image, min_shape_size)
        output_path = os.path.join(output_dir_masks2, filename)

        cv2.imwrite(output_path, colored_image)

    shutil.rmtree(output_dir_masks)

    input_dir_masks2 = output_dir_masks2
    output_dir_list = os.path.join(output_dir, "list_sep")
    if not os.path.exists(output_dir_list):
        os.makedirs(output_dir_list)

    file_list = os.listdir(input_dir_masks2)
    image_objects_count = {}

    for filename in file_list:
        input_path = os.path.join(input_dir_masks2, filename)
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        object_count = extract_objects(image, output_dir_list, filename)
        image_objects_count[filename] = object_count

    with open(os.path.join(output_dir_list, "summary.txt"), "w") as summary_file:
        for filename, count in image_objects_count.items():
            summary_file.write(f"{filename} : {count} objets\n")

# Exemple d'utilisation :
locate("input/fish2_mp4")
