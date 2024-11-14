from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

X = df
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

feature_importances = model.feature_importances_

importances_df = pd.DataFrame({
    'feature': data.feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importances_df, palette="viridis")
plt.title("RandomForestRegressor")
plt.xlabel("Важность")
plt.ylabel("Переменные")
plt.show()
