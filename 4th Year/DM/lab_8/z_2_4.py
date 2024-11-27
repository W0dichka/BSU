import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import umap

mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

y = y.astype(int)
umap_model = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_model.fit_transform(X)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=y, palette="Set2", s=10, alpha=0.6)
plt.title("UMAP: 2D projection of MNIST")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(title="Digit", loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
