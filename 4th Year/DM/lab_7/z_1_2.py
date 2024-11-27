import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

correlation_matrix = wine_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица признаков Iris')
plt.show()

threshold = 0.8

high_correlation_var = np.where(abs(correlation_matrix) > threshold)

to_drop = set()

for i in range(len(high_correlation_var[0])):
    if high_correlation_var[0][i] != high_correlation_var[1][i]: 
        colname = wine_df.columns[high_correlation_var[1][i]]
        to_drop.add(colname)

wine_reduced_df = wine_df.drop(columns=to_drop)

print("Исключенные признаки:", to_drop)
print("Остальные признаки:")
print(wine_reduced_df.head())


correlation2_matrix = wine_reduced_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation2_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Корреляционная матрица признаков Iris')
plt.show()