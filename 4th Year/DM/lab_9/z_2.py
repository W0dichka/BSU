import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

data = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
X = data.drop("medv", axis=1)  
y = data["medv"]


model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_values = []  

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

mean_mse = np.mean(mse_values)

print("MSE на каждом фолде:", mse_values)
print("Среднее значение MSE:", mean_mse)
