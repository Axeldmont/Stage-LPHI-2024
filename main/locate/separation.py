import os
import cv2
import numpy as np

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

input_dir = "masks2"
output_dir = "list"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
file_list = os.listdir(input_dir)
image_objects_count = {}

for filename in file_list:
    input_path = os.path.join(input_dir, filename)
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    object_count = extract_objects(image, output_dir, filename)
    image_objects_count[filename] = object_count

with open(os.path.join(output_dir, "summary.txt"), "w") as summary_file:
    for filename, count in image_objects_count.items():
        summary_file.write(f"{filename} : {count} objets\n")
