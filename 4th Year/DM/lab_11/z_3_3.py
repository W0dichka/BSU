import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle

coordinates = [
    (53.9, 27.5667),  # Минск
    (52.0975, 23.7333),  # Брест
    (52.4417, 30.9875),  # Гомель
    (53.6667, 23.8),  # Гродно
    (53.9, 30.3333),  # Могилев
    (55.1833, 30.2),  # Витебск
    (53.15, 29.1833),  # Бобруйск
    (53.1333, 26.0167),  # Барановичи
    (52.1206, 26.0856),  # Пинск
    (54.5167, 30.4167),  # Орша
    (55.4764, 28.5978),  # Новополоцк
    (52.9, 27.5333),  # Слуцк
    (53.8833, 25.2833),  # Лида
    (52.9500, 30.0333),  # Жлобин
    (40.7128, -74.0060), # Нью-Йорк
    (48.8566, 2.3522),   # Париж
    (52.0444, 29.2497),  # Мозырь
]

coords = np.array(coordinates)

def haversine(x, y):
    return great_circle(x, y).km

db = DBSCAN(eps=200, min_samples=2, metric=haversine)
db.fit(coords)
labels = db.labels_
print("Кластеры:", labels)
for coord, label in zip(coordinates, labels):
    print(f"Координаты: {coord}, Кластер: {label}")
