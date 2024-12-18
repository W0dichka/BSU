import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

file_path = "customers.csv"  
data = pd.read_csv(file_path)
print(data.head())
data = data.drop(columns=["Channel", "Region"], errors='ignore')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

# --- Кластеризация K-Means ---

cluster_range = range(1, 11)
inertia_values = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(scaled_features)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Сумма квадратов расстояний')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)
data["Cluster"] = kmeans_labels
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
data["PCA1"] = pca_features[:, 0]
data["PCA2"] = pca_features[:, 1]

plt.figure(figsize=(10, 7))
for cluster in range(num_clusters):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}", s=30)

plt.title("K-Means Clustering of Wholesale Customers Data")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.show()

cluster_counts = data["Cluster"].value_counts()
print("\nКоличество клиентов в каждом кластере:")
print(cluster_counts)

cluster_means = data.groupby("Cluster").mean()
print("\nСредние значения характеристик для каждого кластера:")
print(cluster_means)