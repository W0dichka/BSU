n = int(input("Введите максимальное число n: "))

possible_numbers = set(range(1, n + 1))

while True:
    try:
        query = input("Введите вопрос и ответ (или пустую строку для завершения): ")
        if not query:
            break
        parts = query.split()
        numbers = set(map(int, parts[:-1]))
        answer = parts[-1]
        
        if answer == "YES":
            possible_numbers &= numbers
        elif answer == "NO":
            possible_numbers -= numbers

    except ValueError:
        print("Ошибка ввода, попробуйте еще раз.")
        continue
        
result = sorted(possible_numbers)
print("Числа, которые мог задумать Август:", " ".join(map(str, result)))