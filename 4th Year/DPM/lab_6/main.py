import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

np.random.seed(52)
p1_x = np.random.normal(2, 1, 100)  
p1_y = np.random.normal(2, 1, 100)
p2_x = np.random.normal(3, 1, 100)
p2_y = np.random.normal(3, 1, 100)

plt.figure(figsize=(8, 6))
plt.scatter(p1_x, p1_y, color='blue', label='Класс 1', alpha=0.7)
plt.scatter(p2_x, p2_y, color='red', label='Класс 2', alpha=0.7)
plt.title("Две группы точек на плоскости", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()



p1 = np.column_stack((p1_x, p1_y))
p1_labels = np.zeros(100)  # Класс 0

p2 = np.column_stack((p2_x, p2_y))
p2_labels = np.ones(100)  # Класс 1


X = np.vstack((p1, p2))
y = np.hstack((p1_labels, p2_labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##############################a##############################################################
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Точность модели:", accuracy)
print("\nМатрица ошибок:\n", conf_matrix)
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Тренировочные данные", alpha=0.6)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', label="Тестовые предсказания", s=80)
plt.title("Наивный байесовский классификатор", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

##############################b##############################################################

model = SVC(kernel='linear', random_state=42) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, label="Тренировочные данные")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=80, label="Тестовые предсказания")
coef = model.coef_[0]
intercept = model.intercept_[0]
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef[0] / coef[1]) * x_vals - (intercept / coef[1])
plt.plot(x_vals, y_vals, color='green', label="Разделяющая гиперплоскость")
plt.title("Метод опорных векторов (SVM)", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()



##############################с##############################################################

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Точность модели (логистическая регрессия):", accuracy)
print("\nМатрица ошибок:\n", conf_matrix)
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, label="Тренировочные данные")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=80, label="Тестовые предсказания")
coef = model.coef_[0]
intercept = model.intercept_[0]
x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_vals = -(coef[0] / coef[1]) * x_vals - (intercept / coef[1])
plt.plot(x_vals, y_vals, color='green', label="Разделяющая гиперплоскость")
plt.title("Логистическая регрессия", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()


##############################d##############################################################

model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Точность модели (деревья решений):", accuracy)
print("\nМатрица ошибок:\n", conf_matrix)
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))


plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, label="Тренировочные данные")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=80, label="Тестовые предсказания")
plt.title("Деревья решений", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=["X1", "X2"], class_names=["Класс 0", "Класс 1"], filled=True)
plt.title("Структура дерева решений", fontsize=14)
plt.show()


##############################e##############################################################

nb_classifier = GaussianNB()
svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
lr_classifier = LogisticRegression(random_state=42)

voting_classifier = VotingClassifier(
    estimators=[
        ('Naive Bayes', nb_classifier),
        ('SVM', svm_classifier),
        ('Logistic Regression', lr_classifier)
    ],
    voting='soft' 
)

voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Точность ансамблевого классификатора:", accuracy)
print("\nМатрица ошибок:\n", conf_matrix)
print("\nОтчет о классификации:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', alpha=0.6, label="Тренировочные данные")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=80, label="Тестовые предсказания")
plt.title("Ансамблевый классификатор (усреднение 3 методов)", fontsize=14)
plt.xlabel("X", fontsize=12)
plt.ylabel("Y", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()






classifiers = {
    "Наивный байес": GaussianNB(),
    "Метод опорных векторов": SVC(kernel='linear', probability=True, random_state=42),
    "Логистическая регрессия": LogisticRegression(random_state=42),
    "Дерево решений": DecisionTreeClassifier(random_state=42, max_depth=3),
}

predictions = {}
for name, model in classifiers.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Класс 0", "Класс 1"])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f"{name}")

plt.tight_layout()
plt.show()

for name, y_pred in predictions.items():
    print(f"Метод: {name}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)



p3_x = np.random.uniform(2, 3.5, 20)
p3_y = np.random.uniform(2, 3.5, 20)
p3 = np.column_stack((p3_x, p3_y))

for name, model in classifiers.items():
    model.fit(X_train, y_train)

p3_predictions = {}
for name, model in classifiers.items():
    p3_predictions[name] = model.predict(p3)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (name, preds) in zip(axes, p3_predictions.items()):
    ax.scatter(p1[:, 0], p1[:, 1], c='blue', label="Класс 0 (p1)", alpha=0.6)
    ax.scatter(p2[:, 0], p2[:, 1], c='red', label="Класс 1 (p2)", alpha=0.6)
    ax.scatter(p3[:, 0], p3[:, 1], c=preds, cmap='coolwarm', edgecolor='k', s=80, label="Точки p3")
    ax.set_title(f"{name}")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()