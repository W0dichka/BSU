import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # Производная сигмоиды
    return 1 / (1 + np.exp(-x))  # Сигмоида

# Входные данные
X = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],  # X
              [1, 0, 1, 0, 1, 0, 1, 0, 1],  # X
              [1, 0, 1, 0, 1, 0, 0, 1, 0],  # Y
              [1, 0, 1, 0, 1, 0, 0, 1, 0],  # Y
              [1, 0, 0, 1, 0, 0, 1, 1, 1],  # L
              [1, 0, 0, 1, 0, 0, 1, 1, 1],  # L
              [1, 1, 1, 0, 1, 0, 1, 1, 1],  # I
              [1, 1, 1, 0, 1, 0, 1, 1, 1]]) # I

# Выходные данные для X, Y, L, I
y = np.array([[1, 0, 0, 0],  # X
              [1, 0, 0, 0],  # X
              [0, 1, 0, 0],  # Y
              [0, 1, 0, 0],  # Y
              [0, 0, 1, 0],  # L
              [0, 0, 1, 0],  # L
              [0, 0, 0, 1],  # I
              [0, 0, 0, 1]]) # I


np.random.seed(1)

# Инициализация весов
syn0 = 2 * np.random.random((9, 4)) - 1  # Входной слой (3 нейрона) к скрытому слою (4 нейрона)
syn1 = 2 * np.random.random((4, 4)) - 1  # Скрытый слой (4 нейрона) к выходному слою (4 нейрона)

# Обучение сети
for j in range(60000):
    # Прямое распространение
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))   # Выходной слой

    # Вычисление ошибки
    l2_error = y - l2
    
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
        
    # Обратное распространение
    l2_delta = l2_error * nonlin(l2, deriv=True)  # Ошибка на выходе
    l1_error = l2_delta.dot(syn1.T)  # Ошибка для первого скрытого слоя
    l1_delta = l1_error * nonlin(l1, deriv=True)  # Ошибка с учетом производной

    # Обновление весов
    syn1 += l1.T.dot(l2_delta)  # Обновление весов между скрытым и выходным слоями
    syn0 += l0.T.dot(l1_delta)  # Обновление весов между входным и скрытым слоями

# Вывод результатов с округлением
print("Выходные данные после тренировки:")
print(np.round(l2,3))


test_cases = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1],  # X
                       [1, 0, 1, 0, 1, 0, 0, 1, 0],  # Y
                       [1, 0, 0, 1, 0, 0, 1, 1, 1],  # L
                       [1, 1, 1, 0, 1, 0, 1, 1, 1],  # I
                       [1, 0, 1, 0, 1, 0, 1, 1, 1],  # X broken
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 1
                       [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
                       [1, 1, 1, 1, 0, 0, 1, 1, 1]]) # L broken

l1_test = nonlin(np.dot(test_cases, syn0))  # Первый скрытый слой
l2_test = nonlin(np.dot(l1_test, syn1))      # Выходной слой

print("Результаты для тестовых случаев:")
for i, result in enumerate(np.round(l2_test, 3)):
    action = np.argmax(result)  # Индекс максимального значения
    if action == 0:
        print(f"Тестовый случай {i+1}: X")
    elif action == 1:
        print(f"Тестовый случай {i+1}: Y")
    elif action == 2:
        print(f"Тестовый случай {i+1}: L")
    else:
        print(f"Тестовый случай {i+1}: I")