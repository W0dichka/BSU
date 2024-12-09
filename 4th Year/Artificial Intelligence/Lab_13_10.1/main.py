import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Устанавливаем параметры
max_features = 5000  # Максимальное количество слов
maxlen = 80          # Максимальная длина последовательности
np.random.seed(42)   # Фиксация случайности

# Загрузка модели
json_file = open("cc_model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# Восстановление структуры модели
loaded_model = model_from_json(loaded_model_json)

# Загрузка весов модели
loaded_model.load_weights("cc_model.weights.h5")  # Убедитесь, что этот файл существует
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Модель успешно загружена")

# Загрузка текста
with open("spam.txt", "r", encoding="utf-8") as file:
    text_data = file.read()

# Обработка текста: используем Tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts([text_data])  # Обучаем токенайзер на тексте

# Преобразуем текст в числовую последовательность
encoded_text = tokenizer.texts_to_sequences([text_data])

# Заполнение последовательности до фиксированной длины
padded_sequence = pad_sequences(encoded_text, maxlen=maxlen)

# Предсказание
prediction = loaded_model.predict(padded_sequence)
print(f"Результат предсказания: {prediction}")
