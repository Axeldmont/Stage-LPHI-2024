from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

def generate_colors(num_colors):
    colors = []
    
    # Générer les couleurs aléatoires pour les composantes verte et bleue
    for _ in range(num_colors - 1):
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        color = (blue, green, 0)
        colors.append(color)
    
    return colors

def locate(path_img):
    # Charger l'image :
    img = cv2.imread(path_img)

    # Convertir l'image en espace de couleur HSV :
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définir les plages HSV pour le rouge :
    lower_red = np.array([0, 70, 70])  # Plage basse pour la teinte, la saturation et la valeur
    upper_red = np.array([10, 255, 255])  # Plage haute pour la teinte, la saturation et la valeur

    # Créer un masque pour les pixels rouges :
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Trouver les contours des zones rouges :
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image pour dessiner les rectangles :
    img_with_boundaries = img.copy()
    img_with_boundaries2 = img.copy()

    cv2.drawContours(img_with_boundaries2, contours, -1, (0,255,0), 3)

    # Dessiner des rectangles autour des zones rouges détectées :
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boundaries, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sauvegarde des images : 
    slic_output_path = path_img.replace('.jpg', '_slic.jpg')
    cv2.imwrite(slic_output_path, img_with_boundaries2)
    bordered_output_path = path_img.replace('.jpg', '_bordered.jpg')
    cv2.imwrite(bordered_output_path, img_with_boundaries)

    return slic_output_path, bordered_output_path

def locatev2(path_img, min_contour_area=1000):
    # Charger l'image :
    img = cv2.imread(path_img)

    # Convertir l'image en espace de couleur HSV :
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définir les plages HSV pour le rouge :
    lower_red = np.array([0, 70, 70])  # Plage basse pour la teinte, la saturation et la valeur
    upper_red = np.array([10, 255, 255])  # Plage haute pour la teinte, la saturation et la valeur

    # Créer un masque pour les pixels rouges :
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Trouver les contours des zones rouges :
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image pour dessiner les rectangles :
    img_with_boundaries = img.copy()
    img_with_boundaries2 = img.copy()

    # Dessiner des contours sur l'image img_with_boundaries2 :
    for i, contour in enumerate(contours):
        # Calculer l'aire du contour
        contour_area = cv2.contourArea(contour)
        # Ignorer les contours trop petits
        if contour_area < min_contour_area:
            continue
        # Générer une couleur aléatoire
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # Dessiner le contour avec la couleur aléatoire
        cv2.drawContours(img_with_boundaries2, [contour], -1, color, 3)
        # Dessiner le rectangle seulement si sa taille dépasse le seuil minimum
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boundaries, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sauvegarde des images :
    slic_output_path = path_img.replace('.jpg', '_slic.jpg')
    cv2.imwrite(slic_output_path, img_with_boundaries2)
    bordered_output_path = path_img.replace('.jpg', '_bordered.jpg')
    cv2.imwrite(bordered_output_path, img_with_boundaries)

    return slic_output_path, bordered_output_path

def locatev3(path_img, min_contour_area=1000):
    # Charger l'image :
    img = cv2.imread(path_img)

    # Convertir l'image en espace de couleur HSV :
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définir les plages HSV pour le rouge :
    lower_red = np.array([0, 70, 70])  # Plage basse pour la teinte, la saturation et la valeur
    upper_red = np.array([10, 255, 255])  # Plage haute pour la teinte, la saturation et la valeur

    # Créer un masque pour les pixels rouges :
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Trouver les contours des zones rouges :
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image pour dessiner les rectangles :
    img_with_boundaries = img.copy()
    img_with_boundaries2 = img.copy()

    # Liste des couleurs disponibles pour chaque contour :
    colors = generate_colors(100)  

    # Dessiner des contours sur l'image img_with_boundaries2 :
    for i, contour in enumerate(contours):
        # Calculer l'aire du contour
        contour_area = cv2.contourArea(contour)
        # Ignorer les contours trop petits
        if contour_area < min_contour_area:
            continue
        # Dessiner le contour seulement si sa taille dépasse le seuil minimum
        cv2.drawContours(img_with_boundaries2, [contour], -1, colors[i % len(colors)], 3)
        # Dessiner le rectangle seulement si sa taille dépasse le seuil minimum
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boundaries, (x, y), (x + w, y + h), colors[i % len(colors)], 2)

    # Sauvegarde des images :
    slic_output_path = path_img.replace('.jpg', '_slic.jpg')
    cv2.imwrite(slic_output_path, img_with_boundaries2)
    bordered_output_path = path_img.replace('.jpg', '_bordered.jpg')
    cv2.imwrite(bordered_output_path, img_with_boundaries)

    return slic_output_path, bordered_output_path

def locatev3vid(frame, min_contour_area=1000):
    # Convertir l'image en espace de couleur HSV :
    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Définir les plages HSV pour le rouge :
    lower_red = np.array([0, 25, 25])  # Plage basse pour la teinte, la saturation et la valeur
    upper_red = np.array([10, 255, 255])  # Plage haute pour la teinte, la saturation et la valeur

    # Créer un masque pour les pixels rouges :
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Trouver les contours des zones rouges :
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image pour dessiner les rectangles :
    img_with_boundaries = frame.copy()
    img_with_boundaries2 = frame.copy()

    # Liste des couleurs disponibles pour chaque contour :
    colors = generate_colors(100)  

    # Dessiner des contours sur l'image img_with_boundaries2 :
    for i, contour in enumerate(contours):
        # Calculer l'aire du contour
        contour_area = cv2.contourArea(contour)
        # Ignorer les contours trop petits
        if contour_area < min_contour_area:
            continue
        # Dessiner le contour seulement si sa taille dépasse le seuil minimum
        cv2.drawContours(img_with_boundaries2, [contour], -1, colors[i % len(colors)], 3)
        # Dessiner le rectangle seulement si sa taille dépasse le seuil minimum
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boundaries, (x, y), (x + w, y + h), colors[i % len(colors)], 2)

    return img_with_boundaries2  # Retourner l'image traitée

def process_video(input_video_path, output_video_path):
    # Ouvrir le fichier vidéo en lecture
    cap = cv2.VideoCapture(input_video_path)
    
    # Vérifier si l'ouverture du fichier vidéo a réussi
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir le fichier vidéo.")
        return
    
    # Récupérer les informations sur les frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Créer un objet VideoWriter pour écrire la vidéo de sortie
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (frame_width, frame_height))
    
    # Processus de traitement de chaque frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Appeler la fonction locatev3 pour détecter les objets et obtenir les contours
            processed_frame = locatev3vid(frame)
            
            # Ajouter la frame traitée à la vidéo de sortie
            out.write(processed_frame)
        else:
            break
    
    # Libérer les ressources
    cap.release()
    out.release()
    cv2.destroyAllWindows()


class Macrophage:
    def __init__(self, contour):
        self.contour = contour
        self.coordinates = self.get_coordinates()

    def get_coordinates(self):
        # Trouver les coordonnées du contour
        coordinates = []
        for point in self.contour:
            coordinates.append((point[0][0], point[0][1]))
        return coordinates

def locatev4(path_img, min_contour_area=1000):
    # Charger l'image :
    img = cv2.imread(path_img)

    # Convertir l'image en espace de couleur HSV :
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Définir les plages HSV pour le rouge :
    lower_red = np.array([0, 70, 70])  # Plage basse pour la teinte, la saturation et la valeur
    upper_red = np.array([10, 255, 255])  # Plage haute pour la teinte, la saturation et la valeur

    # Créer un masque pour les pixels rouges :
    mask = cv2.inRange(img_hsv, lower_red, upper_red)

    # Trouver les contours des zones rouges :
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une liste pour stocker les macrophages détectés
    macrophages = []

    # Dessiner des contours sur l'image :
    for contour in contours:
        # Calculer l'aire du contour
        contour_area = cv2.contourArea(contour)
        # Ignorer les contours trop petits
        if contour_area < min_contour_area:
            continue
        # Créer une instance de la classe Macrophage pour enregistrer les coordonnées du contour
        macrophage = Macrophage(contour)
        macrophages.append(macrophage)

    # Créer une copie de l'image pour dessiner les rectangles :
    img_with_boundaries = img.copy()

    # Dessiner des contours sur l'image img_with_boundaries :
    for macrophage in macrophages:
        cv2.drawContours(img_with_boundaries, [np.array(macrophage.coordinates)], -1, [255, 0, 0], 3)

    # Sauvegarde de l'image avec les contours :
    contour_output_path = path_img.replace('.jpg', '_contour.jpg')
    cv2.imwrite(contour_output_path, img_with_boundaries)

    return macrophages


############################## VIDEO ##################################


class Macrophage:
    def __init__(self, frame_number, contour: list) -> None:
        self.frame_number = frame_number
        self.contour = contour
        self.coordinates = self.get_coordinates()

    def get_coordinates(self):
        coordinates = []
        for point in self.contour:
            coordinates.append((point[0][0], point[0][1]))
        return coordinates

def locatev4(video_path, min_contour_area=500, save_frames_with_contours=False):
    cap = cv2.VideoCapture(video_path)
    frames = []
    macrophages_per_frame = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    for frame_number, frame in enumerate(frames):
        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 25, 25])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(img_hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        macrophages = []
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            if contour_area < min_contour_area:
                continue
            macrophage = Macrophage(frame_number, contour)
            macrophages.append(macrophage)
            if save_frames_with_contours:
                cv2.drawContours(frame, [contour], -1, [0, 255, 0], 3)
        macrophages_per_frame.append(macrophages)

        if save_frames_with_contours:
            cv2.imwrite(f"frame_{frame_number}_with_contours.jpg", frame)

    cap.release()

    return macrophages_per_frame