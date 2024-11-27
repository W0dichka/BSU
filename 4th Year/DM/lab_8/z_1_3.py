import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

def apply_pca(data, n_components=8):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return explained_variance, principal_components

explained_variance_raw, _ = apply_pca(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
explained_variance_scaled, _ = apply_pca(X_scaled)

minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X)
explained_variance_minmax, _ = apply_pca(X_minmax)

normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
explained_variance_normalized, _ = apply_pca(X_normalized)

plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(explained_variance_raw), label='Без предобработки', marker='o')
plt.plot(np.cumsum(explained_variance_scaled), label='Масштабирование (StandardScaler)', marker='o')
plt.plot(np.cumsum(explained_variance_minmax), label='Нормализация (MinMaxScaler)', marker='o')
plt.plot(np.cumsum(explained_variance_normalized), label='Нормализация (Normalizer)', marker='o')
plt.xlabel('Число компонент')
plt.ylabel('Суммарная объяснённая дисперсия')
plt.title('Сравнение влияния предобработки на результаты PCA (California Housing)')
plt.legend()
plt.grid()
plt.show()
