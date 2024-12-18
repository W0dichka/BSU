import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()
penguins['species'] = penguins['species'].astype('category').cat.codes
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
X = penguins[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== Кластеризация ======

cluster_range = range(1, 11)
inertia_values = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Сумма квадратов расстояний')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

hierarchical = linkage(X_scaled, method='ward')
dendrogram(hierarchical, truncate_mode='level', p=3)
plt.title('Иерархическая кластеризация (дендрограмма)')
plt.show()

hierarchical_labels = fcluster(hierarchical, t=3, criterion='maxclust')

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)


def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='Set1', legend="full")
    plt.title(title)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.show()

plot_clusters(X_scaled, kmeans_labels, 'K-means Clustering')
plot_clusters(X_scaled, hierarchical_labels, 'Hierarchical Clustering')
plot_clusters(X_scaled, dbscan_labels, 'DBSCAN Clustering')

kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)

print(f"Silhouette Score для K-means: {kmeans_silhouette:.2f}")
print(f"Silhouette Score для иерархической кластеризации: {hierarchical_silhouette:.2f}")

unique_labels = set(dbscan_labels)
print(f"Количество кластеров, найденных DBSCAN (включая шум): {len(unique_labels)}")
