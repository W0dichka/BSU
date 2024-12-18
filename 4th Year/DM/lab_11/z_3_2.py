import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis', marker='o')
plt.title('Результаты кластеризации с DBSCAN (Moons)')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Метки кластеров')
plt.show()
print("Уникальные метки кластеров DBSCAN:", np.unique(y_dbscan))
