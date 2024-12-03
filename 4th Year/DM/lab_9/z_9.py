import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

X, y = make_regression(n_samples=500, n_features=100, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=20, step=5)
selector.fit(X_train, y_train)

X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

selected_features = np.where(selector.support_)[0]
print("Отобранные признаки:", selected_features)

ridge_full = Ridge(alpha=1.0).fit(X_train, y_train)
lasso_full = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)

ridge_selected = Ridge(alpha=1.0).fit(X_train_selected, y_train)
lasso_selected = Lasso(alpha=0.1, max_iter=10000).fit(X_train_selected, y_train)

y_pred_ridge_full = ridge_full.predict(X_test)
y_pred_lasso_full = lasso_full.predict(X_test)

y_pred_ridge_selected = ridge_selected.predict(X_test_selected)
y_pred_lasso_selected = lasso_selected.predict(X_test_selected)

metrics = {
    "Model": ["Ridge (All)", "Lasso (All)", "Ridge (Selected)", "Lasso (Selected)"],
    "R²": [
        r2_score(y_test, y_pred_ridge_full),
        r2_score(y_test, y_pred_lasso_full),
        r2_score(y_test, y_pred_ridge_selected),
        r2_score(y_test, y_pred_lasso_selected),
    ],
    "MSE": [
        mean_squared_error(y_test, y_pred_ridge_full),
        mean_squared_error(y_test, y_pred_lasso_full),
        mean_squared_error(y_test, y_pred_ridge_selected),
        mean_squared_error(y_test, y_pred_lasso_selected),
    ],
}


metrics_df = pd.DataFrame(metrics)
print(metrics_df)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(ridge_full.coef_, label="Ridge (All Features)", alpha=0.7)
plt.plot(ridge_selected.coef_, label="Ridge (Selected Features)", alpha=0.7)
plt.title("Ridge Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lasso_full.coef_, label="Lasso (All Features)", alpha=0.7)
plt.plot(lasso_selected.coef_, label="Lasso (Selected Features)", alpha=0.7)
plt.title("Lasso Coefficients")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.legend()

plt.tight_layout()
plt.show()
