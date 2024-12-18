import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

iris = load_iris()
data = iris.data
labels_true = iris.target
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
linkage_matrix = linkage(data_scaled, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Дендрограмма")
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel("Образцы")
plt.ylabel("Евклидово расстояние")
plt.show()

num_clusters = 3
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
ari_score = adjusted_rand_score(labels_true, clusters)
print(f"ARI (Adjusted Rand Index) между кластерами и метками: {ari_score:.2f}")

plt.figure(figsize=(8, 6))
for cluster in np.unique(clusters):
    plt.scatter(data_scaled[clusters == cluster, 0],
                data_scaled[clusters == cluster, 1],
                label=f'Кластер {cluster}')

plt.title("Распределение кластеров (Иерархическая кластеризация)")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.legend()
plt.show()
