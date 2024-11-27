import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE

data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

params = [
    {'n_iter': 250, 'learning_rate': 200},
    {'n_iter': 1000, 'learning_rate': 100},
    {'n_iter': 1000, 'learning_rate': 200},
    {'n_iter': 2500, 'learning_rate': 50}
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, param in enumerate(params):
    tsne = TSNE(n_components=2, random_state=42, n_iter=param['n_iter'], learning_rate=param['learning_rate'])
    X_tsne = tsne.fit_transform(X)
    ax = axes[i]
    for j, target_name in enumerate(target_names):
        ax.scatter(X_tsne[y == j, 0], X_tsne[y == j, 1], label=target_name)
    
    ax.set_title(f"t-SNE (n_iter={param['n_iter']}, learning_rate={param['learning_rate']})")
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.legend()

plt.tight_layout()
plt.show()
