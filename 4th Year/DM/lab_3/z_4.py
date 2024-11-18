import pandas as pd
import pycountry

data = pd.read_csv('GlobalCancerdata.csv', encoding='unicode_escape')

print(data.head())
print(data.info())

country_column = 'ï»¿Women'
countries_in_data = data[country_column].unique()

all_countries = {country.name for country in pycountry.countries}
countries_in_data_set = set(countries_in_data)

missing_countries = all_countries - countries_in_data_set

# Вывод результатов
print(f'Количество стран с данными: {len(countries_in_data_set)}')
print(f'Количество отсутствующих стран: {len(missing_countries)}')
print(f'Отсутствующие страны:\n {missing_countries}')
