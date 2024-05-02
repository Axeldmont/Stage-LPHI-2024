import os
import cv2
import numpy as np
from kartezio.inference import ModelPool
from kartezio.fitness import FitnessIOU
from kartezio.dataset import read_dataset
from numena.image.basics import image_normalize

# Création du dossier masks s'il n'existe pas
output_dir = "masks"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fitness = FitnessIOU()
ensemble = ModelPool(f"./models", fitness, regex="*/elite.json").to_ensemble()
dataset = read_dataset(f"./dataset", counting=True)
p_test = ensemble.predict(dataset.test_x)

for i in range(130):
    mask_list = [image_normalize(pi[0][i]["mask"]) for pi in p_test]
    heatmap = np.array(mask_list).mean(axis=0)
    heatmap_cp = (heatmap * 255.0).astype(np.uint8)

    cv2.imwrite(os.path.join(output_dir, f"heatmap_test_{i}.png"), heatmap_cp)

