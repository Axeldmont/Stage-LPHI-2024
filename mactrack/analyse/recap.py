import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_data(file_path):
    data = pd.read_excel(file_path, index_col=0)
    return data

def calculate_intensity_features(intensity_data):
    def find_peaks_and_prominences(row):
        peaks, properties = find_peaks(row, prominence=0.3)
        prominences = properties['prominences']
        return peaks, prominences
    
    def mean_distance_between_peaks(peaks):
        if len(peaks) > 1:
            distances = np.diff(peaks)
            return distances.mean()
        return np.nan
    
    peaks_and_prominences = intensity_data.apply(lambda row: find_peaks_and_prominences(row), axis=1)
    peaks_list = peaks_and_prominences.apply(lambda x: x[0])
    prominences_list = peaks_and_prominences.apply(lambda x: x[1])
    num_peaks = peaks_list.apply(len)
    mean_prominence = prominences_list.apply(lambda x: x.mean() if len(x) > 0 else 0)
    mean_distance = peaks_list.apply(lambda peaks: mean_distance_between_peaks(peaks) if len(peaks) > 2 else np.nan)
    
    return num_peaks, mean_prominence, mean_distance


def calculate_mean(data):
    return data.mean(axis=1)

def count_valid_entries(data):
    num_valid_entries = data.notna().sum(axis=1)
    return num_valid_entries

def plot_intensity_curves(intensity_data, valid_entry_counts, threshold=10):
    filtered_data = intensity_data[valid_entry_counts > threshold]
    output_folder = "output/plot"
    for index, row in filtered_data.iterrows():
        plt.plot(row)
        plt.xlabel('Temps')
        plt.ylabel('Intensité')
        plt.title(f'Courbe d\'intensité pour l\'entrée {index}')
        plt.savefig(f'{output_folder}/intensity_curve_{index}.png', format='png')
        plt.close()

def aggregate(distance_file, intensity_file, size_file, perimeter_file):
    output_file = "output/data/data.xlsx"
    distance_data = load_data(distance_file)
    intensity_data = load_data(intensity_file)
    size_data = load_data(size_file)
    perimeter_data = load_data(perimeter_file)

    num_peaks, mean_prominence, mean_freq = calculate_intensity_features(intensity_data)
    mean_distance = calculate_mean(distance_data)
    mean_size = calculate_mean(size_data)
    mean_perimeter = calculate_mean(perimeter_data)
    valid_entry_counts = count_valid_entries(intensity_data)

    aggregated_data = pd.DataFrame({
        'peaks': num_peaks,
        'amplitude': mean_prominence,
        'frequence': mean_freq,
        'distance': mean_distance,
        'size': mean_size,
        'perimeter': mean_perimeter,
        'validity': valid_entry_counts,
    })

    aggregated_data.to_excel(output_file, engine='openpyxl')
    print(f"Les données agrégées ont été enregistrées dans {output_file}")
    plot_intensity_curves(intensity_data, valid_entry_counts)
    print(f"Les courbes d'intensité ont été enregistrées pour les entrées valides")