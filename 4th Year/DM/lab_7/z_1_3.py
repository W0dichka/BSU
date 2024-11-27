import pandas as pd
from sklearn.datasets import fetch_openml


boston = fetch_openml(data_id=531, as_frame=True)
boston_df = pd.DataFrame(data=boston.data, columns=boston.feature_names)
boston_df['MEDV'] = boston.target 

correlation_with_target = boston_df.corr()['MEDV'].drop('MEDV')

n = 5

top_n_features = correlation_with_target.abs().nlargest(n)

print("Наибольшие абсолютные значения коэффициентов корреляции:")
print(top_n_features)