import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = newsgroups.data
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
data_tfidf = vectorizer.fit_transform(data).toarray()
linkage_matrix = linkage(data_tfidf, method='ward')

plt.figure(figsize=(10, 7))
plt.title("Дендрограмма")
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.xlabel("Образцы")
plt.ylabel("Евклидово расстояние")
plt.show()

num_clusters = 20
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

silhouette_avg = silhouette_score(data_tfidf, clusters)
print(f"Средний коэффициент силуэта: {silhouette_avg:.2f}")
