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

covariance_matrix = returns.cov()

# Построение эффективного портфеля
num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(len(data.columns))
    weights /= np.sum(weights)  
    weights_record.append(weights)
    portfolio_return = np.dot(weights, expected_return)
    results[0, i] = portfolio_return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    results[1, i] = portfolio_std_dev

portfolios = pd.DataFrame(weights_record, columns=data.columns)
portfolios['Expected Return'] = results[0]
portfolios['Risk (Std Dev)'] = results[1]

max_return_index = portfolios['Expected Return'].idxmax()
most_efficient_portfolio = portfolios.loc[max_return_index]

print("Самый эффективный портфель:")
print(most_efficient_portfolio)

plt.figure(figsize=(12, 6))
plt.scatter(results[1, :], results[0, :], c=results[0, :], cmap='viridis', marker='o')
plt.title('Эффективная граница портфелей')
plt.xlabel('Риск (σ)')
plt.ylabel('Ожидаемая доходность (μ)')
plt.colorbar(label='Ожидаемая доходность (μ)')
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(results[1, :]**2, results[0, :], c=results[0, :], cmap='viridis', marker='o')
plt.title('Эффективная граница портфелей (μ, σ²)')
plt.xlabel('Риск (σ²)')
plt.ylabel('Ожидаемая доходность (μ)')
plt.colorbar(label='Ожидаемая доходность (μ)')
plt.grid()
plt.tight_layout()
plt.show()