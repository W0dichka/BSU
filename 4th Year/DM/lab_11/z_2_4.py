import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler

file_path = "seeds.txt"
columns = ['area', 'perimeter', 'compactness', 'length_of_kernel', 'width_of_kernel', 'asymmetry_coefficient', 'length_of_groove', 'class']
df = pd.read_csv(file_path, header=None, names=columns, delim_whitespace=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.iloc[:, :-1])
Z = sch.linkage(scaled_data, method='ward')
plt.figure(figsize=(10, 7))
sch.dendrogram(Z, labels=df['class'].values, leaf_rotation=90, leaf_font_size=12)
plt.title('Дендрограмма иерархической кластеризации')
plt.xlabel('Образцы')
plt.ylabel('Расстояние')
plt.show()
