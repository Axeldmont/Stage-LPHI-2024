import cv2
import numpy as np
import os

def trace_lines_between_contours(contour_a, contour_b, image_shape, max_line_length):
    # Créer une image vide pour stocker les lignes tracées
    traced_lines_image = np.zeros(image_shape, dtype=np.uint8)

    # Convertir les contours en listes de points
    contour_a = contour_a.squeeze()
    contour_b = contour_b.squeeze()

    # Tracer des lignes entre chaque point de contour_a et contour_b
    for point_a in contour_a:
        for point_b in contour_b:
            line_length = np.linalg.norm(np.array(point_a) - np.array(point_b))
            # Vérifier si la ligne dépasse la taille maximale autorisée
            if line_length <= max_line_length:
                # Tracer la ligne seulement si elle est inférieure ou égale à la taille maximale
                cv2.line(traced_lines_image, tuple(point_a), tuple(point_b), (255, 255, 255), 1)

    return traced_lines_image

def create_mask_from_contour(contour, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    return mask

def paint_black_area_from_mask(image_c, mask):
    # Convertir l'image c en niveaux de gris
    image_c_gray = cv2.cvtColor(image_c, cv2.COLOR_BGR2GRAY)
    
    # Inverser le masque (pour extraire la zone noire)
    inverted_mask = cv2.bitwise_not(mask)
    
    # Extraire la zone noire de l'image c
    black_area = cv2.bitwise_and(image_c_gray, image_c_gray, mask=inverted_mask)
    
    # Peindre la zone noire en noir sur l'image c
    image_c_black = cv2.merge((black_area, black_area, black_area))
    
    return image_c_black

def extract_and_save_objects(image_path, output_dir, min_object_size=100):
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
        
        if not replaced:
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

def process_images(image_a_path, image_b_path, image_c_path, max_line_length=50):
    # Charger les images A, B et C
    image_a = cv2.imread(image_a_path, cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(image_b_path, cv2.IMREAD_GRAYSCALE)
    image_c = cv2.imread(image_c_path)
    print(image_a)
    # Trouver les contours des images A et B
    contours_a, _ = cv2.findContours(image_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_b, _ = cv2.findContours(image_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une image pour stocker les résultats
    result_image = np.zeros_like(image_a)

    # Pour chaque contour de l'image A
    for contour_a in contours_a:
        # Pour chaque contour de l'image B
        for contour_b in contours_b:
            # Tracer les lignes entre les contours et obtenir l'image résultante
            traced_lines_image = trace_lines_between_contours(contour_a, contour_b, image_a.shape, max_line_length)
            # Ajouter l'image résultante à l'image finale
            result_image = cv2.bitwise_or(result_image, traced_lines_image)

    # Créer des masques noirs à partir des contours trouvés sur les images A et B
    mask_a = np.zeros_like(image_a)
    for contour_a in contours_a:
        mask_a = cv2.bitwise_or(mask_a, create_mask_from_contour(contour_a, image_a.shape))

    mask_b = np.zeros_like(image_b)
    for contour_b in contours_b:
        mask_b = cv2.bitwise_or(mask_b, create_mask_from_contour(contour_b, image_b.shape))

    # Peindre les zones délimitées par les contours sur l'image résultante en noir
    result_image_with_masks = cv2.bitwise_and(result_image, 255 - mask_a)
    result_image_with_masks = cv2.bitwise_and(result_image_with_masks, 255 - mask_b)

    # Peindre la zone noire obtenue sur l'image c
    image_c = paint_black_area_from_mask(image_c, result_image_with_masks)

    # Enregistrer l'image c modifiée
    cv2.imwrite(image_c_path, image_c)

    output_dir = os.path.dirname(image_c_path)
    extract_and_save_objects(image_c_path, output_dir)

def calculate_iou(segmentation1, segmentation2):
    intersection = np.logical_and(segmentation1, segmentation2)
    union = np.logical_or(segmentation1, segmentation2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def defuse(n=3):

    c = 0

    for i in range (1,n):
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
                process_images(list[0],list[1],image_path)
            c = 0

defuse()