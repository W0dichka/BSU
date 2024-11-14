import pandas as pd

data = {
    'Уровень образования': ['Высшее', 'Среднее', 'Неполное высшее', 'Среднее', 'Высшее'],
    'Удовлетворенность работой': ['Низкая', 'Средняя', 'Высокая', 'Средняя', 'Высокая']
}

df = pd.DataFrame(data)

def transform_data(df, categorical_columns):
    for column in categorical_columns:
        if column == 'Уровень образования':
            categories = ['Неполное высшее', 'Среднее', 'Высшее']
            df[column] = pd.Categorical(df[column], categories=categories, ordered=True)
        elif column == 'Удовлетворенность работой':
            categories = ['Низкая', 'Средняя', 'Высокая']
            df[column] = pd.Categorical(df[column], categories=categories, ordered=True)
    return df

transformed_df = transform_data(df, ['Уровень образования', 'Удовлетворенность работой'])

print(transformed_df)
print(transformed_df.dtypes)
