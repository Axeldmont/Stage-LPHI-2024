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
    if len(image_c.shape) == 2: 
        image_c_gray = image_c
    elif len(image_c.shape) == 3: 
        image_c_gray = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unsupported image format. Expected grayscale (1 channel) or BGR (3 channels).")

    inverted_mask = cv2.bitwise_not(mask)
    black_area = cv2.bitwise_and(image_c_gray, image_c_gray, mask=inverted_mask)
    image_c_black = cv2.merge((black_area, black_area, black_area))
    
    return image_c_black

def extract_and_save_objects(image_c, image, heatmap, object, image_storage, min_object_size=100):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    replaced = False
    n = len(image_storage.get_list(f"heatmap_test_{heatmap}"))
    j = 0
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < min_object_size:
            continue     
        mask = np.zeros_like(image_gray)
        object_image = cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
        object_image = cv2.bitwise_and(image_c, image_c, mask=mask)
        if not replaced :
            image_storage.replace_image(f"heatmap_test_{heatmap}", f"object_{object}.png", object_image)
            replaced = True
        else:
            image_storage.add_image(f"heatmap_test_{heatmap}", f"object_{n+j}.png", object_image)
            j = j + 1
    return image_storage


def process_images(list, image_c, heatmap, object,image_storage,  max_line_length=50):
    n = len(list)
    print (n)
    traced_lines_image = trace_lines_between_contours(list)
    result_image = np.zeros_like(traced_lines_image)
    result_image = cv2.bitwise_or(result_image, traced_lines_image)

    for i in range(n):
        contours, _ = cv2.findContours(list[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_image, contours, -1, (0, 0, 0), thickness=cv2.FILLED)

    image = paint_black_area_from_mask(image_c, result_image)
    
    extract_and_save_objects(image_c, image, heatmap, object, image_storage)
    return image_storage

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2).sum()
    union = np.logical_or(image1, image2).sum()
    return intersection / union if union != 0 else 0

def save_segmentation_images(segmentation_instance, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for dir_name, objects_list in segmentation_instance.images:
        heatmap_folder = os.path.join(output_folder, dir_name)
        os.makedirs(heatmap_folder, exist_ok=True)

        for file_name, sparse_image in objects_list:
            dense_image = sparse_image.toarray().astype(np.uint8)
            image_path = os.path.join(heatmap_folder, file_name)
            cv2.imwrite(image_path, dense_image)

def defuse(n, image_storage):
    c = 0
    for i in range(1, n):
        print(f"heatmap_test_{i}")
        current = len(image_storage.get_list(f"heatmap_test_{i}"))
        for j in range(current):
            image1 = image_storage.get_image(f"heatmap_test_{i}", f"object_{j}.png")
            if image1 is not None:
                image1 = image1.toarray()
                _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
            else : 
                print(i,j)

            matches = []
            previous = len(image_storage.get_list(f"heatmap_test_{i-1}"))
            for k in range(previous):
                image2 = image_storage.get_image(f"heatmap_test_{i-1}", f"object_{k}.png")
                if image2 is not None:
                    image2 = image2.toarray()
                    _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
                else:
                    print(i-1,k)
                iou = calculate_iou(image1, image2)
                if iou > 0:
                    matches.append(image2)
                    c += 1
            if c > 1:
                print(i,j)
                image_storage = process_images(matches, image1, i, j, image_storage)
            c = 0
    return image_storage


def invdefuse(n, image_storage):
    c = 0
    for i in range(1, n):
        print(f"heatmap_test_{n-i-1}")
        current = len(image_storage.get_list(f"heatmap_test_{n-i-1}"))
        for j in range(current):
            image1 = image_storage.get_image(f"heatmap_test_{n-i-1}", f"object_{j}.png")
            if image1 is not None:
                image1 = image1.toarray()
                _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
            else : 
                print(n-i-1,j)

            matches = []
            previous = len(image_storage.get_list(f"heatmap_test_{n-i}"))
            for k in range(previous):
                image2 = image_storage.get_image(f"heatmap_test_{n-i}", f"object_{k}.png")
                if image2 is not None:
                    image2 = image2.toarray()
                    _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)
                else:
                    print(n-i,k)
                iou = calculate_iou(image1, image2)
                if iou > 0:
                    matches.append(image2)
                    c += 1
            if c > 1:
                print(n-i-1,j)
                image_storage = process_images(matches, image1, n-i-1, j, image_storage)
            c = 0
    save_segmentation_images(image_storage, "output/list_def")
    return image_storage