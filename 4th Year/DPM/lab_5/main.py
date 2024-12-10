from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

np.random.seed(52)

def generate_cluster(center_x, center_y, num_points, spread):
    x_coords = np.random.normal(loc=center_x, scale=spread, size=num_points)
    y_coords = np.random.normal(loc=center_y, scale=spread, size=num_points)
    return np.column_stack((x_coords, y_coords))

num_clusters = 5 
points_per_cluster = [40, 50, 30, 60, 40]
spread_values = [1.0, 1.5, 0.8, 1.2, 1.0] 

cluster_centers = [(2, 3), (8, 10), (5, 5), (12, 1), (7, -4)]

points = []
for i in range(num_clusters):
    cluster_points = generate_cluster(
        center_x=cluster_centers[i][0],
        center_y=cluster_centers[i][1],
        num_points=points_per_cluster[i],
        spread=spread_values[i]
    )
    points.append(cluster_points)

points = np.vstack(points)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.7)
plt.title('Множество точек на плоскости')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.grid(True)
plt.show()

cluster_range = range(1, 11)
inertia_values = []

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia_values, marker='o', linestyle='--', color='b')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Сумма квадратов расстояний')
plt.xticks(cluster_range)
plt.grid(True)
plt.show()



k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(points)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))

for cluster_id in range(k):
    cluster_points = points[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)

plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Центры кластеров')

plt.title('Кластеризация методом k-средних')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)
plt.show()



X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=0.2, random_state=42)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for cluster_id in range(k):
    cluster_points = X_train[y_train == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тренировочная выборка')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for cluster_id in range(k):
    cluster_points = X_test[y_test == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тестовая выборка')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





knn = KNeighborsClassifier(n_neighbors=4) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели на тестовой выборке: {accuracy:.2f}\n")
print("Отчет о классификации:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for cluster_id in range(k):
    cluster_points = X_train[y_train == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тренировочная выборка')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for cluster_id in range(k):
    cluster_points = X_test[y_pred == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Предсказанный Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тестовая выборка (KNN)')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()






y_train_pred = knn.predict(X_train)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for cluster_id in range(k):
    cluster_points = X_train[y_train == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тренировочные данные (истинные метки)')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for cluster_id in range(k):
    cluster_points = X_train[y_train_pred == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Кластер {cluster_id + 1}', alpha=0.6)
plt.title('Тренировочные данные (предсказанные метки)')
plt.xlabel('X координата')
plt.ylabel('Y координата')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

conf_matrix = confusion_matrix(y_train, y_train_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[f'Кластер {i+1}' for i in range(k)])
disp.plot(cmap='Blues', values_format='d')
plt.title('Матрица ошибок для тренировочных данных')
plt.show()

conf_matrix_df = pd.DataFrame(
    conf_matrix,
    columns=[f'Предсказано: Кластер {i+1}' for i in range(k)],
    index=[f'Истинное: Кластер {i+1}' for i in range(k)]
)

print("Сводная таблица результатов классификации:")
print(conf_matrix_df)