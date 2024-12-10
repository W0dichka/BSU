import numpy as np
import matplotlib.pyplot as plt

# Параметры
r = 0.05  # Базовая ставка (5%)
n_years = 30  # Горизонт расчетов

# Платежные потоки
cashflows_annuity = [0] * 14 + [1000] * (n_years - 14)  # Аннуитет
cashflows_life = [0, 0, 0, 0, 10000] + [0] * (n_years - 5)  # Страхование дожития

# Суммарный поток пассивов
cashflows_liabilities = [a + l for a, l in zip(cashflows_annuity, cashflows_life)]

# Расчет PV, дюрации и выпуклости без дисконтирования
def calculate_metrics_without_discounting(cashflows):
    k = np.arange(1, len(cashflows) + 1)  # Периоды
    pv = np.sum(cashflows)  # Текущая стоимость
    duration = np.sum(k * np.array(cashflows)) / pv  # Дюрация
    convexity = np.sum(k * (k + 1) * np.array(cashflows)) / pv  # Выпуклость
    return pv, duration, convexity

pv_liabilities, duration_liabilities, convexity_liabilities = calculate_metrics_without_discounting(cashflows_liabilities)

# Облигации
bond_cashflows = {
    "5y": [8] * 4 + [108] + [0] * (n_years - 5),  # 8% купон, погашение через 5 лет
    "15y": [8] * 14 + [108] + [0] * (n_years - 15),  # 8% купон, погашение через 15 лет
    "30y": [0] * 29 + [100],  # Без купона, погашение через 30 лет
}

# Расчет для облигаций
bond_metrics = {}
for bond, cashflows in bond_cashflows.items():
    bond_metrics[bond] = calculate_metrics_without_discounting(cashflows)

# Система уравнений для иммунизации
A = np.array([
    [1, 1, 1],  # Сумма долей должна быть 1
    [bond_metrics["5y"][1], bond_metrics["15y"][1], bond_metrics["30y"][1]],  # Дюрации
    [bond_metrics["5y"][2], bond_metrics["15y"][2], bond_metrics["30y"][2]],  # Выпуклости
])
b = np.array([1, duration_liabilities, convexity_liabilities])  # Нормализуем PV = 1

# Решение системы
x = np.linalg.solve(A, b)
x_5y, x_15y, x_30y = x  # Доли активов в 5-, 15-, и 30-летних облигациях

# Проверка PV, Duration, Convexity для активов
pv_assets = np.dot(x, [bond_metrics["5y"][0], bond_metrics["15y"][0], bond_metrics["30y"][0]])
duration_assets = np.dot(x, [bond_metrics["5y"][1], bond_metrics["15y"][1], bond_metrics["30y"][1]])
convexity_assets = np.dot(x, [bond_metrics["5y"][2], bond_metrics["15y"][2], bond_metrics["30y"][2]])

# Сценарии изменения процентной ставки (New York Seven)
delta_rates = np.linspace(-0.02, 0.02, 7)  # Изменения ставки от -2% до +2%

# Функция для оценки изменений PV
def calculate_surplus(delta_r, pv_assets, duration_assets, convexity_assets):
    pv_change_assets = -duration_assets * delta_r * pv_assets + 0.5 * convexity_assets * (delta_r ** 2) * pv_assets
    pv_change_liabilities = -duration_liabilities * delta_r * pv_liabilities + 0.5 * convexity_liabilities * (delta_r ** 2) * pv_liabilities
    return pv_change_assets - pv_change_liabilities

# Расчет сюрплюса
surpluses = [calculate_surplus(delta_r, pv_assets, duration_assets, convexity_assets) for delta_r in delta_rates]

print(x_5y)
print(x_15y)
print(x_30y)