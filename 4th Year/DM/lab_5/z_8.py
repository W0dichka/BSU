import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("Tweets.csv")
print("Исходные данные:")
print(data.head())

def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
print("\nДанные после очистки текста:")
print(data['cleaned_text'].head())

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
print("\nДанные после удаления стоп-слов:")
print(data['cleaned_text'].head())

# Лемматизация
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()]))
print("\nДанные после лемматизации:")
print(data['cleaned_text'].head())

# Векторизация
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_text'])
print("\nРазмерность векторизованных данных:", X.shape)
print("Пример векторизованных данных (первая строка):")
print(X[0])
