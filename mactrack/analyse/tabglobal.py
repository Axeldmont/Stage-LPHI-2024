import pandas as pd
import os

def tabglobal(result_path):
    results = []
    for folder_name in os.listdir(result_path):
        folder_path = os.path.join(result_path, folder_name)

        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'data', 'data.xlsx')

            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                num_time_rows = df['Time'].notna().sum()
                num_peaks_positive = (df['peaks'] > 0).sum()
                sum_peaks = df['peaks'].sum()
                results.append({
                    'folder': folder_name,
                    'num_time_rows': num_time_rows,
                    'num_peaks_positive': num_peaks_positive,
                    'sum_peaks': sum_peaks
                })

    results_df = pd.DataFrame(results)
    results_df.to_excel('results_summary.xlsx', index=False)
