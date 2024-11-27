import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

model = LogisticRegression(max_iter=200)

rfe = RFE(estimator=model, n_features_to_select=2)

pipeline = Pipeline(steps=[('feature_selection', rfe), ('classification', model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
print("Accuracy scores for each fold: ", scores)
print("Mean accuracy: ", np.mean(scores))