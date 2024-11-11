from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from keras.datasets import cifar100
from keras.utils import to_categorical

# Количество классов изображений
nb_classes = 100

# Загрузка модели из JSON-файла
with open("cifar100_model.json", "r") as json_file:
    model_json = json_file.read()
    
# Десериализация JSON в модель
loaded_model = model_from_json(model_json)

# Загружаем веса в модель
loaded_model.load_weights("cifar100_model.weights.h5")

# Компилируем модель
loaded_model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])

print("Модель успешно загружена и скомпилирована.")

# Загружаем данные для тестирования модели
(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

# Нормализуем данные
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Преобразуем метки в категории
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

# Оцениваем точность модели на тестовых данных
scores = loaded_model.evaluate(X_test, Y_test, verbose=2)
print("Точность модели на тестовых данных: %.2f%%" % (scores[1] * 100))

# Загрузка и обработка изображения для предсказания
im = Image.open('car.png')
im = im.resize((32, 32))  # Изменяем размер изображения на 32x32
im = im.convert('RGB')  # Преобразуем изображение в RGB
im_array = np.array(im).astype('float32') / 255  # Преобразуем в массив и нормализуем
im_array = np.expand_dims(im_array, axis=0)  # Добавляем измерение для батча

# Предсказание класса изображения
prediction = loaded_model.predict(im_array)
print(prediction)
predicted_class = np.argmax(prediction, axis=1)

# Печатаем результат
print("Предсказанный класс:", predicted_class[0])