import cv2
import numpy as np
import os

def trace_lines_between_contours(images, distance_threshold=50):
    traced_lines_image = np.zeros_like(images[0])
    all_contours = []

    for image in images:
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)

    for i in range(len(all_contours)):
        for j in range(i + 1, len(all_contours)):
            cnt1 = all_contours[i]
            cnt2 = all_contours[j]
            for point1 in cnt1:
                for point2 in cnt2:
                    dist = np.linalg.norm(point1 - point2)
                    if dist < distance_threshold:
                        pt1 = tuple(point1[0])
                        pt2 = tuple(point2[0])
                        cv2.line(traced_lines_image, pt1, pt2, (255), 1)

    return traced_lines_image

def paint_black_area_from_mask(image_c, mask):
    image_c_gray = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    inverted_mask = cv2.bitwise_not(mask)
    black_area = cv2.bitwise_and(image_c_gray, image_c_gray, mask=inverted_mask)
    image_c_black = cv2.merge((black_area, black_area, black_area))
    
    return image_c_black

def extract_and_save_objects(image_path, output_dir, min_object_size=600):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_image = cv2.imread(image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    replaced = False
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_object_size:
            continue     
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        object_image = cv2.bitwise_and(original_image, original_image, mask=mask)
        if not replaced :
            object_image_path = image_path
            replaced = True
        else:
            j = 0
            while True:
                object_image_path = os.path.join(output_dir, f'object_{j}.png')
                if not os.path.exists(object_image_path):
                    break
                j = j + 1
        
        cv2.imwrite(object_image_path, object_image)

def process_images(list, image_c_path, max_line_length=50):
    n = len(list)
    images = []
    for i in range(n):
        image = cv2.imread(list[i], cv2.IMREAD_GRAYSCALE)
        images.append(image)

    image_c = cv2.imread(image_c_path)  
    traced_lines_image = trace_lines_between_contours(images)
    result_image = np.zeros_like(traced_lines_image)
    result_image = cv2.bitwise_or(result_image, traced_lines_image)
    for i in range(n):
        contours, _ = cv2.findContours(images[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    image_c = paint_black_area_from_mask(image_c, result_image)
    cv2.imwrite(image_c_path, image_c)
    output_dir = os.path.dirname(image_c_path)
    extract_and_save_objects(image_c_path, output_dir)

def calculate_iou(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def defuse():
    c = 0
    for i in range (1,130):
        input_folder_i = f"output/list_sep/heatmap_test_{i}"
        files_i = os.listdir(input_folder_i)
        print(input_folder_i)
        for j in range(0,len(files_i)):
            image_path = os.path.join(input_folder_i, f"object_{j}.png")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # cher
            _, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            input_folder_k = f"output/list_sep/heatmap_test_{i-1}"
            files_k = os.listdir(input_folder_k)
            list = []
            for k in range(0,len(files_k)):
                imagecomp_path = os.path.join(input_folder_k, f"object_{k}.png")
                imagecomp = cv2.imread(imagecomp_path, cv2.IMREAD_GRAYSCALE)
                _, image2 = cv2.threshold(imagecomp, 127, 255, cv2.THRESH_BINARY)
                iou = calculate_iou(image1, image2)
                if iou > 0:
                    list.append(imagecomp_path)
                    c = c + 1
            if c > 1 :
                process_images(list,image_path)
            c = 0

def invdefuse():
    c = 0
    for i in range (1,130):
        input_folder_i = f"output/list_sep/heatmap_test_{129-i}"
        files_i = os.listdir(input_folder_i)
        print(input_folder_i)
        for j in range(0,len(files_i)):
            image_path = os.path.join(input_folder_i, f"object_{j}.png")
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # cher
            _, image1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            input_folder_k = f"output/list_sep/heatmap_test_{i+1}"
            files_k = os.listdir(input_folder_k)
            list = []
            for k in range(0,len(files_k)):
                imagecomp_path = os.path.join(input_folder_k, f"object_{k}.png")
                imagecomp = cv2.imread(imagecomp_path, cv2.IMREAD_GRAYSCALE)
                _, image2 = cv2.threshold(imagecomp, 127, 255, cv2.THRESH_BINARY)
                iou = calculate_iou(image1, image2)
                if iou > 0:
                    list.append(imagecomp_path)
                    c = c + 1       
            if c > 1 :
                process_images(list,image_path)
            c = 0
