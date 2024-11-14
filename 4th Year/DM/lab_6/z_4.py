from sklearn.datasets import load_wine
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  

mutual_info = mutual_info_regression(df, y, random_state=42)

mutual_info_df = pd.DataFrame({
    'feature': data.feature_names,
    'mutual_info': mutual_info
}).sort_values(by='mutual_info', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='mutual_info', y='feature', data=mutual_info_df, palette="viridis")
plt.title("Wine")
plt.xlabel("Mutual Information Score")
plt.ylabel("Переменные")
plt.show()
