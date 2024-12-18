import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

movies_path = "movies.csv"
ratings_path = "ratings.csv"

movies = pd.read_csv(movies_path, low_memory=False)
ratings = pd.read_csv(ratings_path)

def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres]
    except:
        return []

movies['genres_list'] = movies['genres'].apply(extract_genres)
movies['id'] = pd.to_numeric(movies['id'], errors='coerce')
movies.dropna(subset=['id'], inplace=True)
movies['id'] = movies['id'].astype(int)
ratings['movieId'] = ratings['movieId'].astype(int)
average_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
average_ratings.rename(columns={'rating': 'average_rating'}, inplace=True)
movies = movies.merge(average_ratings, left_on='id', right_on='movieId', how='inner')
movies = movies[['title', 'genres_list', 'average_rating']]
movies = movies[movies['genres_list'].map(len) > 0]
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movies['genres_list']), columns=mlb.classes_)
movies = pd.concat([movies, genres_encoded], axis=1)
features = ['average_rating'] + list(mlb.classes_)
data = movies[features]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
valid_indices = ~np.isnan(data_scaled).any(axis=1) & ~np.isinf(data_scaled).any(axis=1)
movies = movies[valid_indices].reset_index(drop=True)
data_scaled = data_scaled[valid_indices]

linked = linkage(data_scaled, method='ward')

plt.figure(figsize=(15, 7))
dendrogram(linked, truncate_mode='lastp', p=30)
plt.title("Дендрограмма фильмов")
plt.xlabel("Samples")
plt.ylabel("Euclidean Distance")
plt.show()

optimal_clusters = 5 
agg_clustering = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
clusters = agg_clustering.fit_predict(data_scaled)
movies['Cluster'] = clusters


print("\nКоличество фильмов в каждом кластере:")
print(movies['Cluster'].value_counts())

numerical_columns = movies.select_dtypes(include=['number']).columns
cluster_analysis = movies.groupby('Cluster')[numerical_columns].mean()

print("\nСредние значения характеристик для каждого кластера:")
print(cluster_analysis)

genre_cluster_distribution = movies.groupby('Cluster')[mlb.classes_].mean()
print("\nЧастота жанров по кластерам:")
print(genre_cluster_distribution)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 7))
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters, cmap='viridis', s=10)
plt.title("Кластеры фильмов (Иерархическая кластеризация)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()