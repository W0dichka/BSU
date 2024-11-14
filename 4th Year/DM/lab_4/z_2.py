import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("weather_description.csv")
data['datetime'] = pd.to_datetime(data['datetime'])
weather_counts = data.iloc[:, 1:].apply(pd.Series.value_counts).fillna(0)
plt.figure(figsize=(12, 8))
sns.heatmap(weather_counts.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

plt.title('Тепловая карта корреляции между параметрами погоды')
plt.show()