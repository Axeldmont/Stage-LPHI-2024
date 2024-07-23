from locate.locate import locate 
from locate.list_sep import segmentation
from video.inputconfig import inputconfig
from track.track import track
from track.filtre import supprimer_petit
from locate.defuse import defuse,invdefuse
from video.result import video,result, videocomp
import time
from line_profiler import LineProfiler
import cv2
import numpy as np
import os
from analyse.intensity import intensity,intensitymed
from analyse.distance import distance
from analyse.size import size
from analyse.perimeter import perimeter
from analyse.recap import aggregate
from analyse.tabglobal import tabglobal

start_time = time.time()
input_folder = "input/control1"
n = 130
#if not os.path.exists("output/data"):
#    os.makedirs("output/data")
#if not os.path.exists("output/plot"):
#    os.makedirs("output/plot")
#frame = inputconfig(input_folder)
#int = intensity(n, frame, input_folder)
#intmed = intensitymed(n,frame, input_folder)
#dis = distance(n)
#siz = size(n)
#per = perimeter(n)
#int = 'output/data/intensity.xlsx'
#intmed = 'output/data/intensitymed.xlsx'
#dis = 'output/data/distance.xlsx'
#siz = 'output/data/size.xlsx'
#per = 'output/data/perimeter.xlsx'
#recap = aggregate(dis,int,intmed,siz,per)

#locate(input_folder)
#image_storage = segmentation("output/list_sep")
#image_storage.load_images()
##
#image_storage = defuse(n, image_storage)
#image_storage = invdefuse(n, image_storage)
#track(n, 0.5, image_storage)
#int = intensity(n, frame, input_folder)
#intmed = intensitymed(n,frame, input_folder)
#dis = distance(n)
#siz = size(n)
#per = perimeter(n)
#recap = aggregate(dis,int,intmed,siz,per)
#supprimer_petit()
#result(input_folder)
#video()
#videocomp()

#tabglobal('results')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Charger les données agrégées
data_file = "results/3hpa_fish3/data/data.xlsx"
data = pd.read_excel(data_file, index_col=0)

# Fonction pour effectuer la régression linéaire et afficher les résultats
def perform_linear_regression(data, x_col, y_col):
    X = data[[x_col]].dropna()
    y = data[y_col].dropna()
    
    # Assurer que les index sont alignés
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Données Réelles')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Régression Linéaire')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Régression Linéaire: {x_col} vs {y_col}\nR² = {r2:.2f}, RMSE = {rmse:.2f}')
    plt.legend()
    plt.show()

    print(f"Régression Linéaire entre {x_col} et {y_col}:")
    print(f" - Coefficient de détermination (R²): {r2:.2f}")
    print(f" - Root Mean Squared Error (RMSE): {rmse:.2f}")


#variables = data.columns
#for i, x_col in enumerate(variables):
#    for y_col in variables[i+1:]:
#       perform_linear_regression(data, x_col, y_col)
#



#########################################################
#profiler = LineProfiler()
#profiler.add_function(defuse)
#profiler.run('defuse(image_storage)')
#profiler.print_stats()


end_time = time.time()  
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds TOTAL")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal, f_oneway

# Chargement des données
data = {
    'folder': [
        '24hpa_fish7', '24hpa_fish8', '3hpa_fish2', '3hpa_fish3', '3hpa_fish4',
        '48hpa_fish10', '48hpa_fish11', '48hpa_fish12', '48hpa_fish13', '48hpa_fish14',
        '48hpa_fish15', '6hpa_fish4', '6hpa_fish5', '6hpa_fish6', '72hpa_fish16',
        '72hpa_fish17', '72hpa_fish18', '72hpa_fish19', '72hpa_fish20', '72hpa_fish21',
        'cont_fish1', 'cont_fish2', 'cont_fish3', 'cont_fish4', 'cont_fish5'
    ],
    'num_time_rows': [
        76, 53, 74, 39, 44, 64, 68, 61, 60, 34, 45, 50, 36, 53, 40, 62, 30, 30, 41, 24,
        41, 40, 19, 36, 41
    ],
    'num_peaks_positive': [
        51, 18, 35, 20, 16, 41, 26, 43, 25, 19, 18, 25, 16, 33, 23, 23, 14, 12, 12, 19,
        19, 15, 12, 14, 19
    ],
    'sum_peaks': [
        232, 46, 105, 64, 27, 128, 62, 258, 98, 43, 43, 80, 42, 84, 78, 91, 59, 43, 30, 64,
        26, 26, 26, 17, 44
    ],
    'group': [
        3, 3, 1, 1, 1, 4, 4, 4, 4, 4, 4, 2, 2, 2, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0
    ]
}


df = pd.DataFrame(data)
df['num_peaks_positive'] = df['num_peaks_positive']/df['num_time_rows']
# Test de normalité
shapiro_results = {col: shapiro(df[col]) for col in ['num_time_rows', 'num_peaks_positive', 'sum_peaks']}
# Test d'homogénéité des variances
levene_results = {col: levene(df[col], df['group']) for col in ['num_time_rows', 'num_peaks_positive', 'sum_peaks']}

print(shapiro_results, levene_results)

# Exécution de l'ANOVA ou du test de Kruskal-Wallis
if all(shapiro_results[col].pvalue > 0.05 for col in shapiro_results) and all(levene_results[col].pvalue > 0.05 for col in levene_results):
    # ANOVA
    anova_results = {
        'num_time_rows': f_oneway(*[df[df['group'] == g]['num_time_rows'] for g in df['group'].unique()]),
        'num_peaks_positive': f_oneway(*[df[df['group'] == g]['num_peaks_positive'] for g in df['group'].unique()]),
        'sum_peaks': f_oneway(*[df[df['group'] == g]['sum_peaks'] for g in df['group'].unique()])
    }
else:
    # Kruskal-Wallis
    kruskal_results = {
        'num_time_rows': kruskal(*[df[df['group'] == g]['num_time_rows'] for g in df['group'].unique()]),
        'num_peaks_positive': kruskal(*[df[df['group'] == g]['num_peaks_positive'] for g in df['group'].unique()]),
        'sum_peaks': kruskal(*[df[df['group'] == g]['sum_peaks'] for g in df['group'].unique()])
    }

print(anova_results) if 'anova_results' in locals() else print(kruskal_results)

plt.figure(figsize=(15, 10))

# Boxplots pour chaque variable
plt.subplot(3, 1, 1)
sns.boxplot(x='group', y='num_time_rows', data=df)
plt.title('Boxplot du Nombre de Time Rows par Groupe')

plt.subplot(3, 1, 2)
sns.boxplot(x='group', y='num_peaks_positive', data=df)
plt.title('Boxplot du Nombre de Peaks Positifs par Groupe')

plt.subplot(3, 1, 3)
sns.boxplot(x='group', y='sum_peaks', data=df)
plt.title('Boxplot du Somme des Pics par Groupe')

plt.tight_layout()
plt.show()
