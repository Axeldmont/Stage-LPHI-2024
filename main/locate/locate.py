from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import numpy as np
import cv2

def locate(path_img):
    # Charger l'image :
    img = cv2.imread(path_img)

    # Algorithme SLIC :
    nb_segments = 1000
    compactness = 10
    superpixel_labels = slic(img, n_segments=nb_segments, compactness=compactness)

    # Repérer les superpixels rouges et les regrouper :
    red_pixels = np.logical_and(img[:, :, 2] > 100, np.logical_and(img[:, :, 1] == 0, img[:, :, 0] == 0))
    superpixels_to_merge = np.unique(superpixel_labels[red_pixels])
    for sp_label in superpixels_to_merge:
        superpixel_labels[superpixel_labels == sp_label] = np.min(superpixels_to_merge)

    # Encadrer en bleu les superpixels rouge :
    img_with_boundaries = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = np.zeros_like(superpixel_labels, dtype=np.uint8)
    for label in superpixels_to_merge:
        mask[superpixel_labels == label] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boundaries, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Sauvegarde des images :
    slic_output_path = path_img.replace('.jpg', '_slic.jpg')
    plt.imsave(slic_output_path, mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), superpixel_labels))
    bordered_output_path = path_img.replace('.jpg', '_bordered.jpg')
    plt.imsave(bordered_output_path, img_with_boundaries)

    return slic_output_path, bordered_output_path