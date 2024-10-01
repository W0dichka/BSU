def power(a, n):
    if n == 0:
        return 1
    elif n > 0:
        result = 1
        for _ in range(n):
            result *= a
        return result
    else:
        return 1 / power(a, -n)

a = float(input("Введите положительное число a: "))
n = int(input("Введите целое число n: "))

result = power(a, n)
check_result = a ** n
print(f"{a}^{n} = {result}")
print(f"Проверка с использованием стандартной функции: {check_result}")