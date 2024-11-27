import pandas as pd
from sklearn.datasets import load_wine

wine = load_wine()
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

wine_df['quality'] = wine.target  

correlation_matrix = wine_df.corr(method='spearman')
correlation_with_target = correlation_matrix['quality'].drop('quality')

n = 5

top_n_features = correlation_with_target.abs().nlargest(n)

# Результаты
print("Наиболее значимые признаки по корреляции Спирмена:")
print(top_n_features)