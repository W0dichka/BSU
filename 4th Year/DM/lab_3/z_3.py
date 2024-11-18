import pandas as pd
from datetime import datetime, timedelta

data = pd.read_csv('covid19_tweets.csv', encoding='unicode_escape')
print(data.head())
print(data.dtypes)

date_column = 'date' 

data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

current_date = pd.Timestamp(datetime.now())
days_back = 5 * 365 
valid_date = current_date - timedelta(days=days_back)

recent_tweets = data[data[date_column] >= valid_date]

print(f'Количество актуальных твитов (за последние {days_back} дней): {len(recent_tweets)}')
print('Актуальные твиты:\n', recent_tweets[[date_column, 'text']]) 
