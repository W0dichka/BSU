import matplotlib.pyplot as plt
import numpy as np

# Данные выборки
X = [5, 1, 3, 2, 5, 2, 3]

# Подсчет абсолютных частот
freqs = {}
for value in X:
    if value in freqs:
        freqs[value] += 1
    else:
        freqs[value] = 1

# Сортировка значений и частот
values = sorted(freqs.keys())
counts = [freqs[value] for value in values]

# Создание полигона абсолютных частот
plt.figure(figsize=(8, 5))
plt.plot(values, counts, marker='o', linestyle='-', color='b')

# Настройка графика
plt.title('Полигон абсолютных частот')
plt.xlabel('Значение')
plt.ylabel('Частота')
plt.xticks(values)
plt.grid()

# Показать график
plt.show()