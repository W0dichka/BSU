import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("USD_Corrected_Month.txt")
data['Date'] = pd.to_datetime(data['Date'])

plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Value'], marker='o')
plt.title('Временной ряд USD')
plt.xlabel('Дата')
plt.ylabel('Значение')
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()