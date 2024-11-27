from sklearn.datasets import load_wine
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt

wine = load_wine()
X = wine.data  
y = wine.target 
target_names = wine.target_names 

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Factor Analysis
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X)


plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, lw=2, label=target_name)
plt.title("PCA of Wine Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.grid()

plt.subplot(1, 2, 2)
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_fa[y == i, 0], X_fa[y == i, 1], color=color, lw=2, label=target_name)
plt.title("Factor Analysis of Wine Dataset")
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.grid()

plt.tight_layout()
plt.show()
