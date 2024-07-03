import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_excel(file_path, index_col=0)
    return data

def calculate_intensity_features(intensity_data):
    mean_intensity = intensity_data.mean(axis=1)
    peaks = (intensity_data.diff(axis=1) > 0).astype(int).diff(axis=1) == -1
    num_peaks = peaks.sum(axis=1)
    return mean_intensity, num_peaks

def calculate_mean(data):
    return data.mean(axis=1)

def plot_histograms(aggregated_data):
    output_folder = "output/plot"
    for column in aggregated_data.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(aggregated_data[column].dropna(), bins=50, edgecolor='black')
        plt.xlabel(column)
        plt.ylabel('Nombre d\'occurrences')
        plt.title(f'Histogramme de {column}')
        plt.savefig(f'{output_folder}/histogram_{column}.png', format='png')
        plt.close()

def count_valid_entries(data):
    num_valid_entries = data.notna().sum(axis=1)
    return num_valid_entries

def aggregate(distance_file, intensity_file,intensitymed_file, size_file, perimeter_file):
    output_file = "output/data/data.xlsx"
    distance_data = load_data(distance_file)
    intensity_data = load_data(intensity_file)
    intensitymed_data = load_data(intensitymed_file)
    size_data = load_data(size_file)
    perimeter_data = load_data(perimeter_file)

    mean_intensity, num_peaks = calculate_intensity_features(intensity_data)
    mean_intensitymed, _ = calculate_intensity_features(intensitymed_data)
    mean_distance = calculate_mean(distance_data)
    mean_size = calculate_mean(size_data)
    mean_perimeter = calculate_mean(perimeter_data)
    valid_entry_counts = count_valid_entries(intensity_data)

    aggregated_data = pd.DataFrame({
        'Intensité Moyenne': mean_intensity,
        'Intensité Median':mean_intensitymed,
        'Nombre de Pics d\'Intensité': num_peaks,
        'Distance Moyenne': mean_distance,
        'Taille Moyenne': mean_size,
        'Périmètre Moyen': mean_perimeter,
        'Nombre de validités': valid_entry_counts
    })

    aggregated_data.to_excel(output_file, engine='openpyxl')
    print(f"Les données agrégées ont été enregistrées dans {output_file}")
    plot_histograms(aggregated_data)
    print(f"Les histogrammes ont été enregistrés")
