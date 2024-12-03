from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np

data = load_diabetes()
X, y = data.data, data.target
feature_names = data.feature_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

ridge_original = Ridge(alpha=1.0)
lasso_original = Lasso(alpha=0.1)
ridge_standard = Ridge(alpha=1.0)
lasso_standard = Lasso(alpha=0.1)
ridge_minmax = Ridge(alpha=1.0)
lasso_minmax = Lasso(alpha=0.1)

ridge_original.fit(X_train, y_train)
lasso_original.fit(X_train, y_train)
ridge_standard.fit(X_train_standard, y_train)
lasso_standard.fit(X_train_standard, y_train)
ridge_minmax.fit(X_train_minmax, y_train)
lasso_minmax.fit(X_train_minmax, y_train)

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Ridge (Original)": ridge_original.coef_,
    "Ridge (Standardized)": ridge_standard.coef_,
    "Ridge (MinMax)": ridge_minmax.coef_,
    "Lasso (Original)": lasso_original.coef_,
    "Lasso (Standardized)": lasso_standard.coef_,
    "Lasso (MinMax)": lasso_minmax.coef_,
})

coefficients = coefficients.round(3)

print(coefficients)

