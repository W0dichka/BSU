from sklearn.datasets import load_wine
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

model = LinearRegression()

selector = RFE(model, n_features_to_select=5)
selector = selector.fit(X, y)

ranking = selector.ranking_
support = selector.support_

rfe_df = pd.DataFrame({
    'feature': X.columns,
    'ranking': ranking,
    'selected': support
}).sort_values(by='ranking')

plt.figure(figsize=(10, 6))
sns.barplot(x='ranking', y='feature', data=rfe_df, palette="viridis", order=rfe_df['feature'])
plt.title("Linear Regression")
plt.xlabel("RFE Ranking")
plt.ylabel("Переменные")
plt.show()

selected_features = rfe_df[rfe_df['selected'] == True]['feature'].tolist()
print("Selected features by RFE:", selected_features)
