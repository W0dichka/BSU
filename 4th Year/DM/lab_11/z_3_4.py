import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Функция для создания синтетических данных в форме мыши
def generate_mouse_data(num_points=1000):
    np.random.seed(42)
    theta = np.linspace(0, 2 * np.pi, num_points // 2)
    x_head = 3 * np.cos(theta) + 5
    y_head = 4 * np.sin(theta) + 6
    theta_ear1 = np.linspace(0, 2 * np.pi, num_points // 4)
    x_ear1 = 1.5 * np.cos(theta_ear1) + 3
    y_ear1 = 1.5 * np.sin(theta_ear1) + 8
    theta_ear2 = np.linspace(0, 2 * np.pi, num_points // 4)
    x_ear2 = 1.5 * np.cos(theta_ear2) + 7
    y_ear2 = 1.5 * np.sin(theta_ear2) + 8
    theta_body = np.linspace(0, 2 * np.pi, num_points // 2)
    x_body = 4 * np.cos(theta_body) + 5
    y_body = 6 * np.sin(theta_body) + 2
    x = np.concatenate([x_head, x_ear1, x_ear2, x_body])
    y = np.concatenate([y_head, y_ear1, y_ear2, y_body])
    x += np.random.normal(0, 0.1, size=x.shape)
    y += np.random.normal(0, 0.1, size=y.shape)
    
    return np.column_stack((x, y))

data = generate_mouse_data(1000)
dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(data)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k')
plt.scatter(data[labels == -1, 0], data[labels == -1, 1], color='red', label='Noise')

plt.title('DBSCAN Clustering on Synthetic Mouse Shape Data')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.show()
