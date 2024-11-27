from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Модель логистической регрессии
log_reg = LogisticRegression(max_iter=200, random_state=42)

rfe = RFE(estimator=log_reg, n_features_to_select=2)
rfe.fit(X_train, y_train)

selected_features = rfe.support_
selected_features_indices = [i for i, x in enumerate(selected_features) if x]

X_train_selected = X_train[:, selected_features_indices]
X_test_selected = X_test[:, selected_features_indices]

log_reg.fit(X_train_selected, y_train)
y_pred_selected = log_reg.predict(X_test_selected)

accuracy_selected = accuracy_score(y_test, y_pred_selected)

log_reg.fit(X_train, y_train)
y_pred_full = log_reg.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)

print(f"Точность модели с отобранными признаками: {accuracy_selected:.4f}")
print(f"Точность модели с полными признаками: {accuracy_full:.4f}")
print(f"Отобранные признаки: {selected_features_indices}")
