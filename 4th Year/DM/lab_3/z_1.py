import pandas as pd


data = pd.read_csv('online_retail_II.csv', encoding='unicode_escape')
print(data.head())

missing_values = data.isnull().sum()
total_values = len(data)
missing_percentage = (missing_values / total_values) * 100

print("Пропущенные значения:\n", missing_values)
print("Процент пропущенных значений:\n", missing_percentage)


overall_missing_percentage = (missing_values.sum() / total_values) * 100
print(f"Общий процент пропущенных значений: {overall_missing_percentage:.2f}%")
