import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

#LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
accuracy_original = accuracy_score(y_test, y_pred)

logreg.fit(X_train_pca, y_train)
y_pred_pca = logreg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

logreg.fit(X_train_lda, y_train)
y_pred_lda = logreg.predict(X_test_lda)
accuracy_lda = accuracy_score(y_test, y_pred_lda)

print(f'Accuracy without dimensionality reduction: {accuracy_original:.4f}')
print(f'Accuracy with PCA: {accuracy_pca:.4f}')
print(f'Accuracy with LDA: {accuracy_lda:.4f}')

plt.figure(figsize=(12, 5))

# PCA
plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis')
plt.title('PCA - Test Set')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')

# LDA
plt.subplot(1, 2, 2)
plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_test, cmap='viridis')
plt.title('LDA - Test Set')
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')

plt.tight_layout()
plt.show()
