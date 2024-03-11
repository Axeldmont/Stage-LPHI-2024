from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Charger l'image
path_img = 'image.jpg'
img = cv2.imread(path_img)

# Paramètres pour l'algorithme SLIC
nb_segments = 3000
compactness = 10

# Appliquer l'algorithme SLIC pour obtenir les superpixels
superpixel_labels = slic(img, n_segments=nb_segments, compactness=compactness)


# Repérer les superpixels rouge à jaune et les regrouper
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        pixel = img[y, x]
        # Vérifier si le pixel est rouge à jaune
        if 10 < pixel[2] <= 255 and pixel[0] == 0 and pixel[1] <= 255:
            # Trouver le superpixel correspondant
            superpixel = superpixel_labels[y, x]
            # Mettre à jour les étiquettes des superpixels voisins
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if 0 <= y + dy < img.shape[0] and 0 <= x + dx < img.shape[1]:
                        neighbor_superpixel = superpixel_labels[y + dy, x + dx]
                        if neighbor_superpixel != superpixel:
                            # Regrouper les superpixels
                            superpixel_labels[superpixel_labels == neighbor_superpixel] = superpixel

# Afficher l'image avec le nouveau découpage en superpixels
plt.figure(figsize=(10, 10))
plt.imshow(mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), superpixel_labels))
plt.title("Nouveau découpage en superpixels")
plt.axis('off')
plt.show()

# Seuil minimal de taille pour les superpixels à considérer
taille_minimale_superpixel = 1000

# Encadrer les superpixels contenant du rouge et jaune et dépassant le seuil minimal de taille
for label in np.unique(superpixel_labels):
    # Extraire les indices des pixels du superpixel
    indices_in_superpixel = np.where(superpixel_labels == label)
    # Vérifier la taille du superpixel
    if len(indices_in_superpixel[0]) >= taille_minimale_superpixel:
        # Extraire les couleurs des pixels correspondants dans l'image originale
        pixels_in_superpixel = img[indices_in_superpixel[0], indices_in_superpixel[1]]
        # Vérifier si le superpixel contient du rouge et jaune
        if np.any((pixels_in_superpixel[:, 2] > pixels_in_superpixel[:, 1]) &
                  (pixels_in_superpixel[:, 1] > pixels_in_superpixel[:, 0])):
            # Trouver les coordonnées des coins du rectangle entourant le superpixel
            y_min, x_min = np.min(indices_in_superpixel, axis=1)
            y_max, x_max = np.max(indices_in_superpixel, axis=1)
            # Encadrer le superpixel avec une bordure bleue
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

# Afficher l'image avec le nouveau découpage en superpixels et les superpixels encadrés
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Superpixels avec bordure bleue (avec seuil minimal de taille)")
plt.axis('off')
plt.show()