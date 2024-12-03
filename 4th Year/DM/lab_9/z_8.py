import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

X, y = make_regression(n_samples=500, n_features=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9] 
alpha = 1.0  

elasticnet_r2_scores = []
elasticnet_weights = {}

for l1_ratio in l1_ratios:
    elasticnet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
    elasticnet.fit(X_train, y_train)
    y_pred = elasticnet.predict(X_test)
    elasticnet_r2_scores.append(r2_score(y_test, y_pred))
    elasticnet_weights[l1_ratio] = elasticnet.coef_

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(l1_ratios, elasticnet_r2_scores, marker='o')
plt.xlabel("L1 Ratio")
plt.ylabel("R²")
plt.title("Зависимость R² от L1 Ratio")
plt.grid(True)

plt.subplot(1, 2, 2)
l1_ratio_last = l1_ratios[-1]
plt.plot(elasticnet_weights[l1_ratio_last], label=f"ElasticNet Coefficients (l1_ratio={l1_ratio_last})")
plt.xlabel("Индекс признака")
plt.ylabel("Вес признака")
plt.title(f"Распределение весов признаков при l1_ratio={l1_ratio_last}")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

coeff_comparison = pd.DataFrame({
    "Feature Index": range(X.shape[1]),
    **{f"L1 Ratio {l1_ratio}": elasticnet_weights[l1_ratio] for l1_ratio in l1_ratios}
}).round(3)

print(coeff_comparison.head(10)) 
