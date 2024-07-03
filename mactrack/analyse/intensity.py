import os
import pandas as pd
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

def graph():
    file_path = 'output/data/intensity.xlsx'
    data = pd.read_excel(file_path)
    data = data.fillna(0)
    individus = data.iloc[:, 0]
    intensites = data.iloc[:, 1:]
    threshold = 0

    for i, individu in enumerate(individus):
        y = intensites.iloc[i]
        x = intensites.columns
        y_masked = np.where(y == threshold, np.nan, y)
    
        plt.plot(x, y_masked, label=individu)
    step = 5
    plt.xticks(intensites.columns[::step], rotation=45)
    plt.xlabel('Temps')
    plt.ylabel('Intensité')
    plt.title('Courbes d\'intensité des individus au cours du temps')
    plt.savefig('output/plot/courbes_intensite_individus.png', format='png')

def graphmed():
    file_path = 'output/data/intensitymed.xlsx'
    data = pd.read_excel(file_path)
    data = data.fillna(0)
    individus = data.iloc[:, 0]
    intensites = data.iloc[:, 1:]
    threshold = 0

    for i, individu in enumerate(individus):
        y = intensites.iloc[i]
        x = intensites.columns
        y_masked = np.where(y == threshold, np.nan, y)
    
        plt.plot(x, y_masked, label=individu)
    step = 5
    plt.xticks(intensites.columns[::step], rotation=45)
    plt.xlabel('Temps')
    plt.ylabel('Intensité')
    plt.title('Courbes d\'intensité des individus au cours du temps')
    plt.savefig('output/plot/courbes_intensitemed_individus.png', format='png')

def calculate_ratio(image_path_contour, image_a, image_f0):
    image_contour = cv2.imread(image_path_contour)
    if image_contour is None:
        raise FileNotFoundError(f"Image not found: {image_path_contour}")
    gray = cv2.cvtColor(image_contour, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image.")
     
    if image_a.shape != image_f0.shape:
        raise ValueError("Image dimensions do not match.")
    green_a = image_a[:, :, 1].astype(np.float64)
    green_f0 = image_f0[:, :, 1].astype(np.float64)

    mask = np.zeros_like(green_a, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_mask = mask[y:y+h, x:x+w]
    cropped_green_a = green_a[y:y+h, x:x+w]
    cropped_green_f0 = green_f0[y:y+h, x:x+w]
    ratio_sum = 0
    pixel_count = 0
 
    for i in range(cropped_mask.shape[0]):
        for j in range(cropped_mask.shape[1]):
            if cropped_mask[i, j] == 255:
                f = cropped_green_a[i, j]
                f_0 = cropped_green_f0[i, j]
                if f_0 != 0:
                    ratio = (f - f_0) / f_0
                    ratio_sum += ratio
                    pixel_count += 1
    
    if pixel_count == 0:
        ratio_mean = 0
    else:
        ratio_mean = ratio_sum / pixel_count

    return ratio_mean

def intensity(n, frame, input_folder):
    base_folder = 'output/list_track'
    output_file = 'output/data/intensity.xlsx'
    if not output_file.endswith('.xlsx'):
        raise ValueError("Output file must have an .xlsx extension")
    image = cv2.imread(os.path.join(input_folder,"vert/moyenne.png"))

    folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith('macrophage_')]
    folders.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    results = []
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        folder_result = {'Time': folder}
        print(folder_path)
        for a in range(0, n):
            found = False

            for file in os.listdir(folder_path):
                if re.match(rf'{a}_\d+\.png', file):
                    found = True
                    input = os.path.join(folder_path,file)
                    x = calculate_ratio(input, frame.frames_v[a - 1], image)
                    folder_result[f'{a}'] = x
                    break
            if not found:
                folder_result[f'{a}'] = 'NA'
        
        results.append(folder_result)
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Report saved to {output_file}")
    graph()
    return output_file

def intensitymed(n, frame, input_folder):
    base_folder = 'output/list_track'
    output_file = 'output/data/intensitymed.xlsx'
    if not output_file.endswith('.xlsx'):
        raise ValueError("Output file must have an .xlsx extension")
    image = cv2.imread(os.path.join(input_folder,"vert/mediane.png"))

    folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d)) and d.startswith('macrophage_')]
    folders.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
    results = []
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        folder_result = {'Time': folder}
        print(folder_path)
        for a in range(0, n):
            found = False

            for file in os.listdir(folder_path):
                if re.match(rf'{a}_\d+\.png', file):
                    found = True
                    input = os.path.join(folder_path,file)
                    x = calculate_ratio(input, frame.frames_v[a - 1], image)
                    folder_result[f'{a}'] = x
                    break
            if not found:
                folder_result[f'{a}'] = 'NA'
        
        results.append(folder_result)
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False, engine='openpyxl')
    print(f"Report saved to {output_file}")
    graphmed()
    return output_file