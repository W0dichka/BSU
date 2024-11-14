import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# 1. SelectKBest
select_k_best = SelectKBest(score_func=f_classif, k=5)
X_train_kbest = select_k_best.fit_transform(X_train, y_train)
X_test_kbest = select_k_best.transform(X_test)
accuracy_kbest = evaluate_model(X_train_kbest, X_test_kbest, y_train, y_test)

# 2. RFE
rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
accuracy_rfe = evaluate_model(X_train_rfe, X_test_rfe, y_train, y_test)

# 3. DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
importances = tree.feature_importances_

indices = np.argsort(importances)[::-1][:5]
X_train_tree = X_train[:, indices]
X_test_tree = X_test[:, indices]
accuracy_tree = evaluate_model(X_train_tree, X_test_tree, y_train, y_test)

print(f"Точность с использованием SelectKBest: {accuracy_kbest:.4f}")
print(f"Точность с использованием RFE: {accuracy_rfe:.4f}")
print(f"Точность с использованием важности признаков (DecisionTree): {accuracy_tree:.4f}")
