import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('creditcard.csv')

#################### z-score

data['z_score'] = np.abs(stats.zscore(data['Amount']))  # Применяем Z-score к столбцу 'Amount'
threshold = 3  #порог 3
anomalies_z = data[data['z_score'] > threshold]
print("Аномалии по Z-score:")
print(anomalies_z)

#################### IQR 

Q1 = data['Amount'].quantile(0.25)
Q3 = data['Amount'].quantile(0.75)
IQR = Q3 - Q1
anomalies_iqr = data[(data['Amount'] < (Q1 - 1.5 * IQR)) | (data['Amount'] > (Q3 + 1.5 * IQR))]
print("Аномалии по IQR:")
print(anomalies_iqr)

plt.figure(figsize=(10, 6))
plt.boxplot(data['Amount'])
plt.title('Boxplot of Transaction Amounts')
plt.ylabel('Transaction Amount')
plt.show()