import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import nltk
import re

file_path = "news.csv"
data = pd.read_csv(file_path)
print(data.head())
data = data[["ID", "TITLE"]]
data = data.drop_duplicates(subset="TITLE")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

data["processed_title"] = data["TITLE"].apply(preprocess_text)

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data["processed_title"])

num_clusters = 5  
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
data["Cluster"] = clusters

pca = PCA(n_components=2)
reduced_features = pca.fit_transform(tfidf_matrix.toarray())
data["PCA1"] = reduced_features[:, 0]
data["PCA2"] = reduced_features[:, 1]

plt.figure(figsize=(10, 7))
for cluster in range(num_clusters):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(cluster_data["PCA1"], cluster_data["PCA2"], label=f"Cluster {cluster}", s=10)

plt.title("K-Means Clustering of News Titles")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.show()

for cluster in range(num_clusters):
    print(f"\nCluster {cluster} Examples:")
    print(data[data["Cluster"] == cluster]["TITLE"].head(5))
