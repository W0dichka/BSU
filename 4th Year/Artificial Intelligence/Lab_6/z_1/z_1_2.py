import numpy as np

def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)  # Производная сигмоиды
    return 1 / (1 + np.exp(-x))  # Сигмоида

# Входные данные для XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Выходные данные для XOR
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# Инициализация весов
syn0 = 2 * np.random.random((2, 4)) - 1  # Входной слой (2 нейрона) к скрытому слою (4 нейрона)
syn1 = 2 * np.random.random((4, 3)) - 1  # Скрытый слой (4 нейрона) к выходному слою (1 нейрон)
syn2 = 2 * np.random.random((3, 1)) - 1

# Обучение сети
for j in range(60000):
    # Прямое распространение
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))  # Первый скрытый слой
    l2 = nonlin(np.dot(l1, syn1))   # Выходной слой
    l3 = nonlin(np.dot(l2, syn2))

    # Вычисление ошибки
    l3_error = y - l3
    
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l3_error))))
        
    # Обратное распространение
    l3_delta = l3_error * nonlin(l3, deriv=True)  
    l2_error = l3_delta.dot(syn2.T) 
    l2_delta = l2_error * nonlin(l2, deriv=True)  # Ошибка на выходе
    l1_error = l2_delta.dot(syn1.T)  # Ошибка для первого скрытого слоя
    l1_delta = l1_error * nonlin(l1, deriv=True)  # Ошибка с учетом производной

    # Обновление весов
    syn2 += l2.T.dot(l3_delta)
    syn1 += l1.T.dot(l2_delta)  # Обновление весов между скрытым и выходным слоями
    syn0 += l0.T.dot(l1_delta)  # Обновление весов между входным и скрытым слоями

# Вывод результатов
print("Выходные данные после тренировки:")
print(l2)