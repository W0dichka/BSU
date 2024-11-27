import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)

plt.figure(figsize=(10, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_lda[y_train == i, 0], X_lda[y_train == i, 1], alpha=0.8, label=target_name)
plt.title('LDA: проекция на первые 2 компоненты')
plt.xlabel('Первая компонента')
plt.ylabel('Вторая компонента')
plt.legend()
plt.grid()
plt.show()

qda = QDA()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)
accuracy_qda = accuracy_score(y_test, y_pred_qda)

print(f"Точность классификации QDA: {accuracy_qda:.2f}")

X_test_lda = lda.transform(X_test)
plt.figure(figsize=(10, 6))
for i, target_name in enumerate(target_names):
    plt.scatter(X_test_lda[y_test == i, 0], X_test_lda[y_test == i, 1], alpha=0.8, label=f"Истинный класс: {target_name}")

plt.scatter(X_test_lda[:, 0], X_test_lda[:, 1], c=y_pred_qda, alpha=0.2, cmap='viridis', marker='x', label="Предсказания (QDA)")
plt.title('QDA: Классифицированные данные на проекции LDA')
plt.xlabel('Первая компонента (LDA)')
plt.ylabel('Вторая компонента (LDA)')
plt.legend()
plt.grid()
plt.show()
