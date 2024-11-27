# Импортируем необходимые библиотеки
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

boston = fetch_openml(name='boston', version=1, as_frame=True)
X = boston.data
y = boston.target
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
feature_importance = pd.DataFrame({'Feature': feature_names})

#Метод 1: Корреляционные методы
correlations = pd.concat([X, y.rename("Target")], axis=1).corr()['Target'].drop('Target').abs()
feature_importance['Correlation'] = correlations.values

#Метод 2: Важность признаков через RandomForest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
feature_importance['RandomForest'] = rf.feature_importances_

#Метод 3: Регуляризация (Lasso)
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
feature_importance['Lasso'] = np.abs(lasso.coef_)

#Метод 4: Мутуальная информация
mutual_info = mutual_info_regression(X_train, y_train, random_state=42)
feature_importance['MutualInfo'] = mutual_info


feature_importance['AverageImportance'] = feature_importance[['Correlation', 'RandomForest', 'Lasso', 'MutualInfo']].mean(axis=1)
feature_importance_sorted = feature_importance.sort_values(by='AverageImportance', ascending=False)

print("Важность признаков:")
print(feature_importance_sorted)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_sorted.melt(id_vars="Feature", 
                                                value_vars=["Correlation", "RandomForest", "Lasso", "MutualInfo"]),
            x="value", y="Feature", hue="variable", palette="viridis")
plt.title("Сравнение важности признаков по различным методам")
plt.xlabel("Значение важности")
plt.ylabel("Признаки")
plt.legend(title="Метод", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

def evaluate_model(selected_features_indices):
    X_train_sel = X_train.iloc[:, selected_features_indices]
    X_test_sel = X_test.iloc[:, selected_features_indices]
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    return r2_score(y_test, y_pred)

top_features_corr = feature_importance_sorted.sort_values(by="Correlation", ascending=False).head(10).index
top_features_rf = feature_importance_sorted.sort_values(by="RandomForest", ascending=False).head(10).index
top_features_lasso = feature_importance_sorted.sort_values(by="Lasso", ascending=False).head(10).index
top_features_mutualinfo = feature_importance_sorted.sort_values(by="MutualInfo", ascending=False).head(10).index

r2_scores = {
    "Correlation": evaluate_model(top_features_corr),
    "RandomForest": evaluate_model(top_features_rf),
    "Lasso": evaluate_model(top_features_lasso),
    "MutualInfo": evaluate_model(top_features_mutualinfo),
}

print("Качество модели (R^@) для различных методов отбора признаков:")
for method, score in r2_scores.items():
    print(f"{method}: {score:.4f}")
