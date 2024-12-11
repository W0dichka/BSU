import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('sales.csv', sep=';', header=0)
data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
data.iloc[:, 1:] = data.iloc[:, 1:].replace(' ', '', regex=True).replace(',', '.', regex=True)
data['Дата'] = pd.to_datetime(data['Дата'], format='%d.%m.%Y')
data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
data.set_index('Дата', inplace=True)

# Рассчёт среднемесячных цен
monthly_average = data.resample('ME').mean() 

# Оценка ожидаемой доходности от инвестиций
returns = (monthly_average.pct_change() * 100).dropna().infer_objects()
expected_return = returns.mean()

# Рассчёт ковариационной матрицы
covariance_matrix = data.cov()

plt.figure(figsize=(12, 6))
for column in monthly_average.columns:
    plt.plot(monthly_average.index, monthly_average[column], label=column)

plt.title('Среднемесячные цены акций')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for column in returns.columns:
    plt.plot(returns.index, returns[column], label=column)

plt.title('Доходность акций')
plt.xlabel('Дата')
plt.ylabel('Доходность (%)')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, 
            xticklabels=covariance_matrix.columns, yticklabels=covariance_matrix.columns)
plt.title('Ковариационная матрица')
plt.show()

print("Среднемесячные цены:")
print(monthly_average)
print("\nОжидаемая доходность от инвестиций за 1 год:")
print(expected_return)
print("\nКовариационная матрица:")
print(covariance_matrix)