import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay

# Загрузка данных
wine = load_wine()
X, y = wine['data'], wine['target']

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Словарь для моделей и их параметров
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr'),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Neural Network': MLPClassifier(max_iter=1000)
}

# Обучение и оценка моделей
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(
        y_test, model.predict_proba(X_test), multi_class='ovr'
    ) if hasattr(model, 'predict_proba') else None
    results[name] = {
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc
    }

# Сравнение точности моделей
for name, metrics in results.items():
    roc_auc_str = f", ROC AUC = {metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else ""
    print(f"{name}: Accuracy = {metrics['accuracy']:.4f}{roc_auc_str}")

# Визуализация матрицы ошибок
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Пример визуализации матрицы ошибок для SVM
plot_confusion_matrix(results['SVM']['confusion_matrix'], title='SVM Confusion Matrix')

# ROC-кривые
plt.figure(figsize=(10, 7))
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        for i in range(len(wine.target_names)):
            fpr, tpr, _ = roc_curve(y_test == i, model.predict_proba(X_test)[:, i])
            plt.plot(fpr, tpr, label=f'{name} - Class {i}')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# Подбор гиперпараметров (пример для SVM)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best parameters for SVM: {grid_search.best_params_}")