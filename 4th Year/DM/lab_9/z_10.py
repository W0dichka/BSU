import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Линейная регрессия
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Huber Regressor
huber_model = HuberRegressor()
huber_model.fit(X_train, y_train)
y_pred_huber = huber_model.predict(X_test)
metrics = {
    "Model": ["Linear Regression", "Huber Regressor"],
    "R²": [
        r2_score(y_test, y_pred_lr),
        r2_score(y_test, y_pred_huber),
    ],
    "MSE": [
        mean_squared_error(y_test, y_pred_lr),
        mean_squared_error(y_test, y_pred_huber),
    ],
}

metrics_df = pd.DataFrame(metrics)
print("Метрики моделей:")
print(metrics_df)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, alpha=0.7, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.title("Linear Regression Predictions")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_huber, alpha=0.7, label="Predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
plt.title("Huber Regressor Predictions")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()

plt.tight_layout()
plt.show()
