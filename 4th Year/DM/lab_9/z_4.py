import pandas as pd
from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X = data.drop("medv", axis=1)  
y = data["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_cv = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 200.0], scoring='neg_mean_squared_error', cv=10)
ridge_cv.fit(X_train, y_train)
ridge_best_alpha = ridge_cv.alpha_
ridge_model = Ridge(alpha=ridge_best_alpha)
ridge_model.fit(X_train, y_train)
ridge_coefficients = ridge_model.coef_

lasso_cv = LassoCV(alphas=np.logspace(-3, 2, 100), cv=10, random_state=42)
lasso_cv.fit(X_train, y_train)
lasso_best_alpha = lasso_cv.alpha_
lasso_model = Lasso(alpha=lasso_best_alpha)
lasso_model.fit(X_train, y_train)
lasso_coefficients = lasso_model.coef_

ridge_zero_weights = np.sum(ridge_coefficients == 0)
lasso_zero_weights = np.sum(lasso_coefficients == 0)
lasso_predictions = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)

print("Ridge: Лучший alpha:", ridge_best_alpha)
print("Ridge: Количество нулевых коэффициентов:", ridge_zero_weights)
print("\nLasso: Лучший alpha:", lasso_best_alpha)
print("Lasso: Количество нулевых коэффициентов:", lasso_zero_weights)
print("Lasso: Mean Squared Error (MSE):", lasso_mse)
