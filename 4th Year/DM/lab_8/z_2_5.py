import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import umap

iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names

umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne_model.fit_transform(X)


# График для UMAP
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="Set2", s=100, alpha=0.7)
plt.title("UMAP: 2D projection of Iris dataset")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Species", loc="center left", bbox_to_anchor=(1, 0.5))

# Подграфик для t-SNE
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="Set2", s=100, alpha=0.7)
plt.title("t-SNE: 2D projection of Iris dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Species", loc="center left", bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()
