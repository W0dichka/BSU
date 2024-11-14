import pandas as pd
import plotly.express as px

data = pd.read_csv('National Assembly 2002 - Updated.csv')
print(data.head())

fig = px.bar(
    data,
    x='Candidate_Name',        # Название кандидата
    y='Votes',                 # Количество голосов
    color='Party',             # Окрас по партии
    title='Голоса по кандидатам',
    hover_data={'Votes': True, 'District': True, 'Constituency_title': True},
    labels={'Votes': 'Количество голосов', 'Candidate_Name': 'Кандидат'}
)

fig.show()
