import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.stats as stats

data = pd.read_csv('avianHabitat.csv')

print("Первоначальные данные:")
print(data.head())

cleaned_data = data[data['WHt'] != 0]


print("\nОчищенные данные:")
print(cleaned_data.head())


# Рассчет статистических показателей
max_wh = cleaned_data['WHt'].max()
min_wh = cleaned_data['WHt'].min()
range_wh = max_wh - min_wh
mean_wh = cleaned_data['WHt'].mean()
median_wh = cleaned_data['WHt'].median()
mode_wh = cleaned_data['WHt'].mode()[0] 
variance_wh = cleaned_data['WHt'].var()
std_dev_wh = cleaned_data['WHt'].std()
first_quartile = cleaned_data['WHt'].quantile(0.25)
third_quartile = cleaned_data['WHt'].quantile(0.75)
iqr = third_quartile - first_quartile
skewness_wh = stats.skew(cleaned_data['WHt'])
kurtosis_wh = stats.kurtosis(cleaned_data['WHt'])

# Вывод результатов
print(f"Максимальное значение высоты растений (WHt): {max_wh:.2f}")
print(f"Минимальное значение высоты растений (WHt): {min_wh:.2f}")
print(f"Размах высоты растений (WHt): {range_wh:.2f}")
print(f"Среднее значение высоты растений (WHt): {mean_wh:.2f}")
print(f"Медиана высоты растений (WHt): {median_wh:.2f}")
print(f"Мода высоты растений (WHt): {mode_wh:.2f}")
print(f"Дисперсия высоты растений (WHt): {variance_wh:.2f}")
print(f"Среднеквадратическое отклонение высоты растений (WHt): {std_dev_wh:.2f}")
print(f"Первый квартиль высоты растений (WHt): {first_quartile:.2f}")
print(f"Третий квартиль высоты растений (WHt): {third_quartile:.2f}")
print(f"Интерквартильный размах высоты растений (WHt): {iqr:.2f}")
print(f"Асимметрия высоты растений (WHt): {skewness_wh:.2f}")
print(f"Эксцесс высоты растений (WHt): {kurtosis_wh:.2f}")



# Построение диаграммы с усами для WHt

plt.figure(figsize=(8, 6))
sns.boxplot(y=cleaned_data['WHt']) 
plt.title('Диаграмма с усами для высоты растений (WHt)')
plt.ylabel('Высота растений (WHt)')
plt.show()


# Построение диаграммы с усами для WHt и EHt

cleaned_data = cleaned_data[cleaned_data['EHt'] != 0]

plt.figure(figsize=(8, 6))
sns.boxplot(data=cleaned_data[['WHt', 'EHt']], orient='v') 
plt.title('Диаграммы с усами для высоты растений (WHt и EHt)')
plt.ylabel('Значение высоты')
plt.xlabel('Переменная')
plt.show()




# Ручное построение эмпирической функции распределения для WHt
sorted_data = np.sort(cleaned_data['WHt'])
n = len(sorted_data)
empirical_cdf_manual = np.arange(1, n+1) / n

ecdf = ECDF(cleaned_data['WHt'])

plt.figure(figsize=(10, 6))

plt.step(sorted_data, empirical_cdf_manual, label='Ручное построение ЭФР', where='post')

plt.plot(sorted_data, ecdf(sorted_data), label='ECDF из statsmodels', linestyle='--')


plt.title('Сравнение ЭФР: ручное построение и ECDF из statsmodels')
plt.xlabel('Высота растений (WHt)')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()



# Построение гистограммы вероятностей и кривой плотности для WHt
plt.figure(figsize=(8, 6))
sns.histplot(cleaned_data['WHt'], stat='density', kde=False, bins=20, color='skyblue', label='Гистограмма')
sns.kdeplot(cleaned_data['WHt'], color='red', label='Кривая плотности (KDE)')

plt.title('Гистограмма вероятностей и кривая плотности для WHt')
plt.xlabel('Высота растений (WHt)')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True)
plt.show()

# Построение QQ-графика для WHt


plt.figure(figsize=(8, 6))
stats.probplot(cleaned_data['WHt'], dist="norm", plot=plt)
plt.title('QQ-Plot для высоты растений (WHt)')
plt.grid(True)
plt.show()
