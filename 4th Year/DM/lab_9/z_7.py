import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = make_regression(n_samples=500, n_features=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.01, 0.1, 1, 10, 100]
ridge_r2_scores = []
lasso_r2_scores = []

ridge_weights = {}
lasso_weights = {}

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_r2_scores.append(r2_score(y_test, y_pred_ridge))
    ridge_weights[alpha] = ridge.coef_
    
    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    lasso_r2_scores.append(r2_score(y_test, y_pred_lasso))
    lasso_weights[alpha] = lasso.coef_

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_r2_scores, label="Ridge R²", marker='o')
plt.plot(alphas, lasso_r2_scores, label="Lasso R²", marker='o')
plt.xscale("log")
plt.xlabel("Alpha (логарифмическая шкала)")
plt.ylabel("R²")
plt.title("Зависимость R² от значения регуляризации")
plt.legend()

plt.subplot(1, 2, 2)
alpha_last = alphas[-1]
plt.plot(ridge_weights[alpha_last], label=f"Ridge Coefficients (alpha={alpha_last})")
plt.plot(lasso_weights[alpha_last], label=f"Lasso Coefficients (alpha={alpha_last})")
plt.xlabel("Индекс признака")
plt.ylabel("Вес признака")
plt.title(f"Распределение весов признаков при alpha={alpha_last}")
plt.legend()

plt.tight_layout()
plt.show()

import pandas as pd

coeff_comparison = pd.DataFrame({
    "Feature Index": range(X.shape[1]),
    f"Ridge Coefficients (alpha={alpha_last})": ridge_weights[alpha_last],
    f"Lasso Coefficients (alpha={alpha_last})": lasso_weights[alpha_last]
}).round(3)

print(coeff_comparison.head(10)) 
