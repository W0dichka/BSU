import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X = data.drop("medv", axis=1)  
y = data["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1, 1.0, 10.0, 100.0, 200.0]

ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=10)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Лучшее значение alpha:", best_alpha)
print("MSE на тестовой выборке:", mse)
