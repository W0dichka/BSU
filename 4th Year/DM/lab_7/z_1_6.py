import pandas as pd
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

data = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Нормализуем данные
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df.drop('target', axis=1)), columns=df.columns[:-1])
df_normalized['target'] = df['target']

# Стандартизируем данные
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df.drop('target', axis=1)), columns=df.columns[:-1])
df_standardized['target'] = df['target']

df_logged = df.copy()
numeric_columns = df_logged.select_dtypes(include=[np.number]).columns
df_logged[numeric_columns] = np.log1p(df_logged[numeric_columns])


corr_original = df.corr()
corr_normalized = df_normalized.corr()
corr_standardized = df_standardized.corr()
corr_logged = df_logged.corr()


def plot_correlation_matrix(corr_matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title(title)
    plt.show()


plot_correlation_matrix(corr_original, 'Оригинальные данные')
plot_correlation_matrix(corr_normalized, 'Нормализованные данные')
plot_correlation_matrix(corr_standardized, 'Стандартизированные данные')
plot_correlation_matrix(corr_logged, 'Логарифмированные данные')


#порог для корреляции
threshold = 0.9

def select_highly_correlated_features(corr_matrix, threshold=0.9):
    correlated_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                correlated_features.add(colname)
    return correlated_features

correlated_features = select_highly_correlated_features(corr_normalized, threshold)
print(f"Сильно коррелированные признаки: {correlated_features}")


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

important_features = X.columns[(lasso.coef_ != 0)]
print(f"Важные признаки после Lasso: {important_features}")







