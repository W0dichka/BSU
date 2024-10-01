import math

def distance(x1, y1, x2, y2):
    try:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    except TypeError:
        raise TypeError("Все аргументы должны быть числами.")

def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Ошибка: пожалуйста, введите действительное число.")

x1 = get_float_input("Введите x1: ")
y1 = get_float_input("Введите y1: ")
x2 = get_float_input("Введите x2: ")
y2 = get_float_input("Введите y2: ")

try:
    result = distance(x1, y1, x2, y2)
    print(f"Расстояние между точками: {result}")
except TypeError as e:
    print(f"Ошибка: {e}")