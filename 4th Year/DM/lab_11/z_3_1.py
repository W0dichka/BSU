import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix, classification_report

iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_dbscan, cmap='viridis', marker='o')
plt.title('Результаты кластеризации с DBSCAN')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Метки кластеров')
plt.show()
print("Сравнение кластеров DBSCAN с реальными метками:")
print(confusion_matrix(y, y_dbscan))
print(classification_report(y, y_dbscan))
