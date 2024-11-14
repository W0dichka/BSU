import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("CAR_DETAILS.csv")  

# 1. Столбчатая диаграмма: Количество автомобилей по типу топлива
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='fuel')
plt.title('Количество автомобилей по типу топлива')
plt.xlabel('Тип топлива')
plt.ylabel('Количество автомобилей')
plt.show()

# 2. Гистограмма: Распределение цен автомобилей
plt.figure(figsize=(10, 6))
sns.histplot(data['selling_price'], bins=30)
plt.title('Распределение цен автомобилей')
plt.xlabel('Цена')
plt.ylabel('Количество')
plt.show()

# 3. Линейный график: Средняя цена автомобиля по годам
plt.figure(figsize=(10, 6))
avg_price_per_year = data.groupby('year')['selling_price'].mean()
plt.plot(avg_price_per_year.index, avg_price_per_year.values, marker='o')
plt.title('Средняя цена автомобиля по годам')
plt.xlabel('Год')
plt.ylabel('Средняя цена')
plt.show()

# 4. Диаграмма размаха (ящик с усами): Цена автомобилей по типу топлива
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='fuel', y='selling_price')
plt.title('Разброс цен автомобилей по типу топлива')
plt.xlabel('Тип топлива')
plt.ylabel('Цена')
plt.show()

# 5. Радиальная диаграмма: Средний пробег для разных типов топлива
avg_mileage_per_fuel = data.groupby('fuel')['km_driven'].mean()
labels = avg_mileage_per_fuel.index
values = avg_mileage_per_fuel.values

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
ax.bar([i * (2 * 3.1415 / len(values)) for i in range(len(values))], values, tick_label=labels)
plt.title('Средний пробег для разных типов топлива')
plt.show()

# 6. Scatter Plot: Зависимость цены от пробега
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='km_driven', y='selling_price')
plt.title('Зависимость цены от пробега')
plt.xlabel('Пробег (км)')
plt.ylabel('Цена (тыс. руб)')
plt.show()


