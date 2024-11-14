import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = pd.read_csv('train.csv')

print(data.head())

#########################

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


data_cleaned = remove_outliers(data, 'SalePrice')

print(f'Original: {data.shape}')
print(f'Cleaned: {data_cleaned.shape}')

data_cleaned['GrLivArea_discrete'] = pd.qcut(data_cleaned['GrLivArea'], q=4, labels=False)
print(data_cleaned[['GrLivArea', 'GrLivArea_discrete']].head())

################
continuous_var = 'SalePrice'
discrete_var = 'BedroomAbvGr'

#Нормальное распределение
mu, std = stats.norm.fit(data[continuous_var])
print(f'Нормальное распределение: mu = {mu:.2f}, std = {std:.2f}')

#Экспоненциальное распределение
lambda_exp = 1 / np.mean(data[continuous_var])
print(f'Экспоненциальное распределение: lambda = {lambda_exp:.2f}')

#Биномиальное распределение
n_binom = 10  # предполагаемое максимальное количество спален
p_binom = np.mean(data[discrete_var]) / n_binom  # приближённое значение вероятности
print(f'Биномиальное распределение: n = {n_binom}, p = {p_binom:.2f}')

#Распределение Пуассона
lambda_poisson = np.mean(data[discrete_var])
print(f'Распределение Пуассона: lambda = {lambda_poisson:.2f}')



# Визуализация непрерывных распределений
plt.figure(figsize=(14, 6))

# Гистограмма данных (нормализованная)
plt.subplot(1, 2, 1)
sns.histplot(data[continuous_var], kde=False, stat='density', bins=30, color='blue', label='Данные')

# Нормальное распределение
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'r-', label=f'Нормальное: mu={mu:.2f}, std={std:.2f}')

# Экспоненциальное распределение
p_exp = lambda_exp * np.exp(-lambda_exp * x)
plt.plot(x, p_exp, 'g-', label=f'Экспоненциальное: lambda={lambda_exp:.2f}')
plt.legend()

plt.title('Приближение непрерывных данных')

# Визуализация дискретных распределений
plt.subplot(1, 2, 2)
sns.histplot(data[discrete_var], kde=False, discrete=True, color='blue', stat='probability', label='Данные')

# Биномиальное распределение
x_binom = np.arange(0, n_binom + 1)
binom_pmf = stats.binom.pmf(x_binom, n_binom, p_binom)

plt.stem(x_binom, binom_pmf, linefmt='r-', markerfmt='ro', basefmt=" ", label=f'Биномиальное: n={n_binom}, p={p_binom:.2f}')

# Распределение Пуассона
x_poisson = np.arange(0, max(data[discrete_var]) + 1)
poisson_pmf = stats.poisson.pmf(x_poisson, lambda_poisson)

plt.stem(x_poisson, poisson_pmf, linefmt='g-', markerfmt='go', basefmt=" ", label=f'Пуассон: lambda={lambda_poisson:.2f}')

plt.legend()

plt.title('Приближение дискретных данных')
plt.tight_layout()
plt.show()

###############################


continuous_var = 'SalePrice'

num_bins = 10
data['binned'] = pd.cut(data[continuous_var], bins=num_bins)

observed_frequencies = data['binned'].value_counts().sort_index()

mu, std = stats.norm.fit(data[continuous_var])

bin_edges = np.histogram_bin_edges(data[continuous_var], bins=num_bins)
expected_frequencies = np.diff(stats.norm.cdf(bin_edges, mu, std)) * len(data)

difference = observed_frequencies.sum() - expected_frequencies.sum()
expected_frequencies[-1] += difference

# Хи-квадрат
chi_square_stat = np.sum((observed_frequencies - expected_frequencies) ** 2 / expected_frequencies)

degrees_of_freedom = num_bins - 1 - 2 

alpha = 0.05
critical_value = stats.chi2.ppf(1 - alpha, degrees_of_freedom)

print(f'Наблюдаемое значение Хи-квадрат: {chi_square_stat:.2f}')
print(f'Критическое значение Хи-квадрат для alpha={alpha} и df={degrees_of_freedom}: {critical_value:.2f}')

if chi_square_stat > critical_value:
    print('Гипотеза о нормальном распределении отвергается.')
else:
    print('Гипотеза о нормальном распределении принимается.')

plt.figure(figsize=(10, 6))
plt.hist(data[continuous_var], bins=num_bins, density=True, alpha=0.6, color='g', label='Данные')

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
plt.plot(x, p, 'r-', label=f'Нормальное распределение: mu={mu:.2f}, std={std:.2f}')

plt.legend()
plt.title('Сравнение данных с нормальным распределением')
plt.show()


############################

print("\nСравнение с встроенной реализацией:")

observed_frequencies_array = observed_frequencies.values
expected_frequencies_array = expected_frequencies

observed_frequencies_array = np.array(observed_frequencies_array, dtype=float)
expected_frequencies_array = np.array(expected_frequencies_array, dtype=float)

chi2_stat_builtin, p_value_builtin = stats.chisquare(f_obs=observed_frequencies_array, f_exp=expected_frequencies_array)

print(f'Наблюдаемое значение Хи-квадрат (встроенное): {chi2_stat_builtin:.2f}')
print(f'P-значение (встроенное): {p_value_builtin:.4f}')

if p_value_builtin < alpha:
    print('Гипотеза о нормальном распределении отвергается.')
else:
    print('Гипотеза о нормальном распределении принимается.')