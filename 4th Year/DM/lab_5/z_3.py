import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('SummaryofWeather.csv')

data['z_score'] = np.abs(stats.zscore(data['MaxTemp']))
threshold = 3  #порог 3
anomalies_z = data[data['z_score'] > threshold]
print("Аномалии по Z-score:")
print(anomalies_z)


Q1 = data['MaxTemp'].quantile(0.25)
Q3 = data['MaxTemp'].quantile(0.75)
IQR = Q3 - Q1

# Определение выбросов
anomalies_iqr = data[(data['MaxTemp'] < (Q1 - 1.5 * IQR)) | (data['MaxTemp'] > (Q3 + 1.5 * IQR))]
print("Аномалии по IQR:")
print(anomalies_iqr)

plt.figure(figsize=(10, 6))
plt.boxplot(data['MaxTemp'])
plt.title('Boxplot of MaxTemp')
plt.ylabel('MaxTemp')
plt.show()