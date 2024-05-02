import os
import cv2
import numpy as np

# Fonction pour filtrer les petites formes
def filter_small_shapes(image, min_size):
    # Recherche des contours dans l'image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Parcours de tous les contours
    for contour in contours:
        # Calcul de la surface du contour
        area = cv2.contourArea(contour)
        
        # Si la surface est inférieure à la taille minimale
        if area < min_size:
            # Colorier la forme en noir
            cv2.drawContours(image, [contour], 0, 0, -1)  # Coloriage en noir
    
    return image

# Chemin du dossier contenant les images masks
input_dir = "masks"

# Création du dossier masks2 s'il n'existe pas
output_dir = "masks2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Liste des fichiers dans le dossier masks
file_list = os.listdir(input_dir)

# Taille minimale des formes à conserver
min_shape_size = 800  # À adapter selon vos besoins

for filename in file_list:
    # Chemin complet de l'image d'entrée
    input_path = os.path.join(input_dir, filename)
    
    # Lecture de l'image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Création d'une image vide de même taille
    colored_image = 255 * (image > 0).astype('uint8')
    
    # Filtrage des petites formes
    colored_image = filter_small_shapes(colored_image, min_shape_size)
    
    # Chemin de sortie pour l'image coloriée
    output_path = os.path.join(output_dir, filename)
    
    # Écriture de l'image coloriée dans le dossier masks2
    cv2.imwrite(output_path, colored_image)
