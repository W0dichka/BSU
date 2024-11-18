import pandas as pd

data = pd.read_csv('GlobalLandTemperaturesByCity.csv', encoding='unicode_escape')

print(data.head())
print(data.dtypes)

date_column = 'dt'  

data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

invalid_dates = data[data[date_column].isnull()]

print(f'Количество некорректных дат: {len(invalid_dates)}')
print('Некорректные даты:\n', invalid_dates)
